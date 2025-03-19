import asyncio
import datetime
import json
import os

import polars as pl
from pytok.tiktok import PyTok
from tqdm import tqdm

hashtag_name = 'romania'

async def main():
    hashtag_df = pl.DataFrame()
    for filename in os.listdir('./data'):
        if filename.endswith('.parquet.zstd') and filename.startswith('hashtag_'):
            hashtag_df = pl.concat([hashtag_df, pl.read_parquet(f'./data/{filename}')], how='diagonal_relaxed')

    hashtag_df = hashtag_df.with_columns(pl.col('author').struct.field('uniqueId').alias('author_id'))
    author_df = hashtag_df.unique('author_id').sort('author_id').drop_nulls('author_id')

    video_path = f'./data/user_videos.parquet.zstd'
    user_path = f'./data/users.parquet.zstd'
    if os.path.exists(video_path):
        video_df = pl.read_parquet(video_path)
        user_df = pl.read_parquet(user_path)
        author_df = author_df.filter(~pl.col('author_id').is_in(user_df['uniqueId']))
    else:
        video_df = pl.DataFrame()
        user_df = pl.DataFrame()

    pbar = tqdm(total=len(author_df))
    async with PyTok(manual_captcha_solves=True) as api:
        for author in author_df['author_id'].to_list():
            user = api.user(username=author)
            user_info = await user.info()

            videos = []
            async for video in user.videos(count=1000):
                video_info = await video.info()
                create_date = datetime.datetime.fromtimestamp(video_info['createTime'])
                if create_date < datetime.datetime(2024, 1, 1):
                    break
                videos.append(video_info)

            try:
                video_df = pl.concat([video_df, pl.DataFrame(videos)], how='diagonal_relaxed')
            except (pl.exceptions.SchemaError, pl.exceptions.PanicException):
                video_df = pl.DataFrame(video_df.to_dicts() + videos, infer_schema_length=len(video_df) + len(videos))
            user_df = pl.concat([user_df, pl.DataFrame([user_info])], how='diagonal_relaxed')
            pbar.update(1)
            video_df.write_parquet(video_path, compression='zstd')
            user_df.write_parquet(user_path, compression='zstd')

if __name__ == "__main__":
    asyncio.run(main())
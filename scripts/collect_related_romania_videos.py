import asyncio
import datetime
import json
import os

import polars as pl
from pytok.tiktok import PyTok
from tqdm import tqdm

from utils import concat

hashtag_name = 'romania'


def filter_romanian(df, keywords):
    df = df.with_columns(pl.col('video').struct.field('subtitleInfos').list.eval(pl.col('').struct.field('LanguageCodeName')).alias('subtitleLanguages'))
    return df.filter(
        pl.col('desc').str.to_lowercase().str.contains_any(keywords)\
        | (pl.col('textLanguage') == 'ro')\
        | ((pl.col('subtitleLanguages').list.contains('ron-RO')) & (pl.col('subtitleLanguages').list.len() < 5))
    )

async def main():
    hashtag_df = pl.DataFrame()
    for filename in os.listdir('./data'):
        if filename.endswith('.parquet.zstd') and filename.startswith('hashtag_'):
            hashtag_df = concat(hashtag_df, pl.read_parquet(f'./data/{filename}'))

    hashtag_df = hashtag_df.unique('id')
    hashtag_df = hashtag_df.with_columns(pl.col('author').struct.field('uniqueId').alias('author_id'))

    keywords = [
        'romania', 'bucharest', 'georgescu', 'lasconi', 'bucuresti', 'iohannis', 'hurezeanu', 'sosoaca', 'ciolacu'\
        'simion', 'nicusor dan', 'bolojan', 'crin antonescu', 'potra', 'ponta', 'mariustuca', 'alegeri'
    ]

    video_path = f'./data/fetched_videos.parquet.zstd'
    related_path = f'./data/related_videos.parquet.zstd'
    if os.path.exists(video_path):
        video_df = pl.read_parquet(video_path)
        related_df = pl.read_parquet(related_path)
        to_fetch_df = concat(hashtag_df, related_df)
        to_fetch_df = to_fetch_df.filter(~pl.col('id').is_in(video_df.select('id')))
    else:
        video_df = pl.DataFrame()
        related_df = pl.DataFrame()
        to_fetch_df = hashtag_df

    pbar = tqdm()
    while len(to_fetch_df) > 0:
        async with PyTok(manual_captcha_solves=False, headless=True) as api:
            try:
                author_id, video_id = to_fetch_df.select(['author_id', 'id']).rows()[0]
                video = api.video(username=author_id, id=video_id)
                video_info = await video.info()
                videos = []
                related_videos = []
                video_info['scrape_date'] = datetime.datetime.today()
                videos.append(video_info)

                async for video_info in video.related_videos():
                    video_info['scrape_date'] = datetime.datetime.today()
                    related_videos.append(video_info)

                video_df = concat(video_df, pl.DataFrame(videos)).unique(subset=['id'])
                related_df = concat(related_df, pl.DataFrame(related_videos)).unique(subset=['id'])
                related_df = related_df.unique(subset=['id'])
                video_df = video_df.unique(subset=['id'])

                to_fetch_df = to_fetch_df.tail(len(to_fetch_df) - 1)
                to_fetch_df = concat(to_fetch_df, related_df).unique(subset=['id'])

                # filter to only videos and related videos that contain keywords in the description
                video_df = filter_romanian(video_df, keywords)
                related_df = filter_romanian(related_df, keywords)
                to_fetch_df = filter_romanian(to_fetch_df, keywords)

                pbar.update(1)
                video_df.write_parquet(video_path, compression='zstd')
                related_df.write_parquet(related_path, compression='zstd')
                print(f"Number videos: {len(video_df)}, Number related videos: {len(related_df)}, Number to fetch: {len(to_fetch_df)}")
            except Exception as e:
                print(e)
                to_fetch_df = to_fetch_df.tail(len(to_fetch_df) - 1)

if __name__ == "__main__":
    asyncio.run(main())
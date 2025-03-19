import asyncio
import json

import polars as pl
from pytok.tiktok import PyTok

hashtag_name = 'romania'

async def main():
    terms = ['romania', 'bucharest', 'georgescu', 'lasconi']

    async with PyTok(manual_captcha_solves=True) as api:
        for term in terms:
            search = api.search(term)

            videos = []
            async for video in search.videos(count=1000):
                video_info = await video.info()
                videos.append(video_info)

            df = pl.DataFrame(videos)
            df.write_parquet(f'./data/{term}.parquet.zstd', compression='zstd')

if __name__ == "__main__":
    asyncio.run(main())
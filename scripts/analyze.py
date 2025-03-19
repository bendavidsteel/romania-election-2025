import os

import polars as pl

def concat(a_df, b_df):
    try:
        return pl.concat([a_df, b_df], how='diagonal_relaxed')
    except (pl.exceptions.SchemaError, pl.exceptions.PanicException):
        return pl.DataFrame(a_df.to_dicts() + b_df.to_dicts(), infer_schema_length=len(a_df) + len(b_df))

def main():
    # df = pl.DataFrame()
    # for filename in os.listdir('./data'):
    #     if filename.endswith('.parquet.zstd'):
    #         try:
    #             df = concat(df, pl.read_parquet(f'./data/{filename}'))
    #         except Exception as e:
    #             print(f"Error processing file {filename}: {e}")

    df = pl.read_parquet('./data/fetched_election_videos.parquet.zstd')

    df = df.unique('id')
    df = df.with_columns(pl.col('desc').str.extract_all(r'#(?P<hashtag>[a-zA-Z0-9_]+)').alias('hashtags'))
    
    # country list
    country_df = df.select(['locationCreated']).drop_nulls()['locationCreated'].value_counts().sort('count')

    # author list
    author_df = df.select([
        pl.col('author').struct.field('uniqueId'), 
        pl.col('author').struct.field('nickname'), 
        pl.col('authorStats').struct.field('followerCount')
        ])\
        .unique('uniqueId')\
        .sort('followerCount')
    
    pass
    

if __name__ == "__main__":
    main()
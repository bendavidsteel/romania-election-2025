import os
import polars as pl
import pycountry
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.dates as mdates
from datetime import datetime

def concat(a_df, b_df):
    try:
        return pl.concat([a_df, b_df], how='diagonal_relaxed')
    except (pl.exceptions.SchemaError, pl.exceptions.PanicException):
        return pl.DataFrame(a_df.to_dicts() + b_df.to_dicts(), infer_schema_length=len(a_df) + len(b_df))

def create_choropleth_maps(df):
    # Load European countries shapefile
    # You'll need to download this or use one you already have
    # A good source would be Natural Earth data: https://www.naturalearthdata.com/
    europe = gpd.read_file('./23686383/Europe/Europe_merged.shp')
    
    # Create a mapping between country names/codes in your data and the shapefile
    # This may need adjustment based on your actual data
    country_mapping = {
        # Map between locationCreated values and country names/codes in shapefile
        # Example: 'GB': 'United Kingdom', 'DE': 'Germany', etc.
    }
    # Map country codes/names in your data to those in the shapefile
    iso_map = {c: pycountry.countries.get(alpha_2=c).alpha_3 for c in df['locationCreated'].unique().to_list()}
    df = df.with_columns(pl.col('locationCreated').replace_strict(iso_map).alias('country_code'))
    
    # Prepare data for choropleth maps
    # 1. Total video count by country
    total_by_country = df.group_by('country_code').agg(pl.count()).sort('count', descending=True)
    total_by_country = total_by_country.rename({'count': 'total_videos'})
    
    # 2. Videos with 'lasconi' in description
    lasconi_df = df.filter(pl.col('desc').str.to_lowercase().str.contains('lasconi', literal=True))
    lasconi_by_country = lasconi_df.group_by('country_code').agg(pl.count()).sort('count', descending=True)
    lasconi_by_country = lasconi_by_country.rename({'count': 'lasconi_videos'})
    
    # 3. Videos with 'georgescu' in description
    georgescu_df = df.filter(pl.col('desc').str.to_lowercase().str.contains('georgescu', literal=True))
    georgescu_by_country = georgescu_df.group_by('country_code').agg(pl.count()).sort('count', descending=True)
    georgescu_by_country = georgescu_by_country.rename({'count': 'georgescu_videos'})
    
    # Convert Polars DataFrames to Pandas for GeoPandas compatibility
    total_pd = total_by_country.to_pandas()
    lasconi_pd = lasconi_by_country.to_pandas()
    georgescu_pd = georgescu_by_country.to_pandas()
    

    # Merge with GeoDataFrame
    total_map = europe.merge(total_pd, left_on='GID_0', right_on='country_code', how='left')
    lasconi_map = europe.merge(lasconi_pd, left_on='GID_0', right_on='country_code', how='left')
    georgescu_map = europe.merge(georgescu_pd, left_on='GID_0', right_on='country_code', how='left')
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot total videos choropleth
    total_map.plot(column='total_videos', cmap='viridis', legend=True, 
                   legend_kwds={'label': 'Total Videos'}, ax=axes[0])
    axes[0].set_title('Total Video Volume by Country')
    axes[0].set_axis_off()
    
    # Plot lasconi videos choropleth
    lasconi_map.plot(column='lasconi_videos', cmap='Reds', legend=True, 
                    legend_kwds={'label': 'Videos with "lasconi"'}, ax=axes[1])
    axes[1].set_title('Videos Mentioning "lasconi" by Country')
    axes[1].set_axis_off()
    
    # Plot georgescu videos choropleth
    georgescu_map.plot(column='georgescu_videos', cmap='Blues', legend=True, 
                      legend_kwds={'label': 'Videos with "georgescu"'}, ax=axes[2])
    axes[2].set_title('Videos Mentioning "georgescu" by Country')
    axes[2].set_axis_off()
    
    plt.tight_layout()
    plt.savefig('europe_video_choropleths.png', dpi=300)
    plt.close()
    
    return total_by_country, lasconi_by_country, georgescu_by_country

def create_time_series(df):
    # Ensure we have a datetime column to work with
    # Assuming your data has a 'createTime' column
    df = df.with_columns(pl.from_epoch(pl.col('createTime').cast(pl.UInt64)).alias('date'))
    
    # Group by date and count videos
    # Daily aggregation
    total_by_date = df.group_by(pl.col('date').dt.date()).agg(pl.count()).sort('date')
    total_by_date = total_by_date.rename({'count': 'total_videos'})
    
    # Filter for lasconi videos and count by date
    lasconi_df = df.filter(pl.col('desc').str.to_lowercase().str.contains('lasconi', literal=True))
    lasconi_by_date = lasconi_df.group_by(pl.col('date').dt.date()).agg(pl.count()).sort('date')
    lasconi_by_date = lasconi_by_date.rename({'count': 'lasconi_videos'})
    
    # Filter for georgescu videos and count by date
    georgescu_df = df.filter(pl.col('desc').str.to_lowercase().str.contains('georgescu', literal=True))
    georgescu_by_date = georgescu_df.group_by(pl.col('date').dt.date()).agg(pl.count()).sort('date')
    georgescu_by_date = georgescu_by_date.rename({'count': 'georgescu_videos'})
    
    # Create time series plot
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    
    # Convert to pandas for easier plotting
    total_pd = total_by_date.to_pandas()
    lasconi_pd = lasconi_by_date.to_pandas()
    georgescu_pd = georgescu_by_date.to_pandas()
    
    # Plot time series
    ax.plot(total_pd['date'], total_pd['total_videos'], 'k-', label='Total Videos', linewidth=2)
    ax.plot(lasconi_pd['date'], lasconi_pd['lasconi_videos'], 'r-', label='Videos with "lasconi"', linewidth=2)
    ax.plot(georgescu_pd['date'], georgescu_pd['georgescu_videos'], 'b-', label='Videos with "georgescu"', linewidth=2)
    
    # Format plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Videos')
    ax.set_title('Video Volume Over Time')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('video_time_series.png', dpi=300)
    plt.close()
    
    return total_by_date, lasconi_by_date, georgescu_by_date

def main():
    # Load data
    # Uncomment the original loading code when you're ready to use it
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
    
    # Create visualizations
    print("Creating choropleth maps...")
    total_country, lasconi_country, georgescu_country = create_choropleth_maps(df)
    
    print("Creating time series plots...")
    total_time, lasconi_time, georgescu_time = create_time_series(df)
    
    # Print some summary statistics
    print("\nTop countries by total video count:")
    print(total_country.head(10))
    
    print("\nTop countries for 'lasconi' videos:")
    print(lasconi_country.head(10))
    
    print("\nTop countries for 'georgescu' videos:")
    print(georgescu_country.head(10))
    
    print("\nVisualizations saved as 'europe_video_choropleths.png' and 'video_time_series.png'")

if __name__ == "__main__":
    main()
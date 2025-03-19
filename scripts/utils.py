import polars as pl

def concat(a_df, b_df):
    try:
        return pl.concat([a_df, b_df], how='diagonal_relaxed')
    except (pl.exceptions.SchemaError, pl.exceptions.PanicException):
        return pl.DataFrame(a_df.to_dicts() + b_df.to_dicts(), infer_schema_length=len(a_df) + len(b_df))

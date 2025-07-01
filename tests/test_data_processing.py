import pandas as pd
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.__init__ import create_aggregate_features, handle_missing

def test_create_aggregate_features():
    test_data = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3],
        'Amount': [10, 20, 30, 40, 50]
    })
    agg = create_aggregate_features(test_data)
    
    assert set(agg.columns) == {'CustomerId', 'TotalAmount', 'AvgAmount', 'StdAmount', 'TxnCount'}
    assert agg.loc[agg['CustomerId'] == 1, 'TotalAmount'].values[0] == 30
    assert agg.loc[agg['CustomerId'] == 3, 'TxnCount'].values[0] == 1

def test_handle_missing():
    df = pd.DataFrame({'A': [1, 2, None, 4]})
    df_filled = handle_missing(df)
    assert df_filled['A'].isnull().sum() == 0

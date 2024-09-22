import pandas as pd

# Example DataFrame with Greater London districts
data = {
    'district': ['BARKING AND DAGENHAM', 'BARNET', 'BEXLEY', 'BRENT', 'BROMLEY', 
                 'CAMDEN', 'CITY OF LONDON', 'CITY OF WESTMINSTER', 'CROYDON', 
                 'EALING', 'ENFIELD', 'GREENWICH', 'HACKNEY', 'HAMMERSMITH AND FULHAM', 
                 'HARINGEY', 'HARROW', 'HAVERING', 'HILLINGDON', 'HOUNSLOW', 
                 'ISLINGTON', 'KENSINGTON AND CHELSEA', 'KINGSTON UPON THAMES', 
                 'LAMBETH', 'LEWISHAM', 'MERTON', 'NEWHAM', 'REDBRIDGE', 
                 'RICHMOND UPON THAMES', 'SOUTHWARK', 'SUTTON', 'TOWER HAMLETS', 
                 'WALTHAM FOREST', 'WANDSWORTH']
}
df = pd.DataFrame(data)

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['district'])

# Display full DataFrame
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_encoded)



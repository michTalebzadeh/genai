import pandas as pd
import numpy as np
from datetime import datetime

# Actual measurements
actual_data = {
    'datetaken': [
        '16/11/2023', '17/11/2023', '26/11/2023', '27/11/2023', '28/11/2023', '01/12/2023',
        '14/12/2023', '15/12/2023', '16/12/2023', '17/12/2023', '18/12/2023', '19/12/2023',
        '21/12/2023', '22/12/2023', '23/12/2023', '24/12/2023', '25/12/2023', '26/12/2023',
        '27/12/2023', '31/12/2023', '01/01/2024', '05/01/2024', '06/01/2024', '07/01/2024',
        '08/01/2024', '09/01/2024', '10/01/2024', '12/01/2024', '13/01/2024', '14/01/2024',
        '15/01/2024', '16/01/2024', '17/01/2024', '18/01/2024', '20/01/2024', '21/01/2024',
        '22/01/2024', '23/01/2024', '26/01/2024', '28/01/2024', '29/01/2024', '31/01/2024',
        '01/02/2024', '04/02/2024', '05/02/2024', '06/02/2024', '07/02/2024', '09/02/2024',
        '11/02/2024', '12/02/2024', '13/02/2024', '14/02/2024', '15/02/2024', '16/02/2024',
        '17/02/2024', '19/02/2024', '20/02/2024', '22/02/2024', '23/02/2024', '24/02/2024',
        '25/02/2024', '26/02/2024', '28/02/2024', '29/02/2024', '01/03/2024', '03/03/2024',
        '04/03/2024', '05/03/2024', '06/03/2024', '07/03/2024', '08/03/2024', '09/03/2024',
        '11/03/2024', '14/03/2024', '15/03/2024', '16/03/2024', '17/03/2024', '18/03/2024',
        '20/03/2024', '22/03/2024', '23/03/2024', '25/03/2024', '26/03/2024', '27/03/2024',
        '29/03/2024', '30/03/2024'
    ],
    'weight': [
        72.2, 72.0, 72.3, 71.2, 71.3, 70.5, 70.0, 70.5, 70.0, 68.5, 68.7, 68.7, 70.0, 70.0, 
        70.2, 70.2, 70.5, 72.0, 72.0, 69.0, 69.8, 69.8, 69.5, 68.5, 67.5, 69.8, 67.9, 68.9, 
        69.5, 69.8, 70.5, 69.5, 68.5, 69.5, 67.7, 69.9, 70.2, 68.2, 69.9, 69.9, 69.9, 70.0, 
        69.9, 69.7, 69.7, 69.2, 69.2, 69.7, 70.4, 69.9, 69.9, 69.9, 70.1, 69.7, 69.7, 70.0, 
        70.1, 70.1, 70.1, 70.5, 70.5, 70.2, 70.2, 68.5, 68.0, 69.5, 70.5, 69.7, 69.9, 67.5, 
        67.5, 67.7, 68.8, 69.4, 70.0, 70.0, 68.0, 68.4, 68.0, 69.2, 69.2, 68.2, 68.9, 69.2, 
        68.2, 68.5
    ]
}

# Convert actual dates to datetime
actual_data['datetaken'] = [datetime.strptime(date, '%d/%m/%Y') for date in actual_data['datetaken']]

# Generate dates
start_date = datetime.strptime('23/10/2023', '%d/%m/%Y')
end_date = datetime.strptime('31/03/2024', '%d/%m/%Y')
date_range = pd.date_range(start_date, end_date)

# Exclude actual measurement dates
actual_dates = set(actual_data['datetaken'])
synthetic_dates = [date for date in date_range if date not in actual_dates]

# Generate synthetic weights
np.random.seed(0)
synthetic_weights = np.round(np.random.uniform(67.5, 72.3, len(synthetic_dates)), 1)

# Create synthetic data DataFrame
synthetic_data = pd.DataFrame({
    'datetaken': synthetic_dates,
    'weight': synthetic_weights
})

# Sort the synthetic data by date
synthetic_data.sort_values('datetaken', inplace=True)

# Convert 'datetaken' to string in the desired format
synthetic_data['datetaken'] = synthetic_data['datetaken'].dt.strftime('%d/%m/%Y')

# Print the synthetic DataFrame
print(synthetic_data.to_string(index=False))


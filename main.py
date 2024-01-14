import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time
import asyncio
import aiohttp
from datetime import datetime

st.set_page_config(
    page_title="DashBoard Farm",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Link API to request data
api_url = 'https://9bd5-115-73-218-247.ngrok-free.app/query'

# Commands SQL to select data
count = 30
table_name = 'sensorParserSmartWater'

# #side bar to do something like add a logo.. maybe i can
# add_selectbox = st.sidebar.selectbox(
#     "Send Notification Via?",
#     ("Email", "Mobile phone")
# )


def apply_scaling_factor(df, sensor):
    if sensor == 'Salinity (‰)':
        return (df['value'].astype(float) * unit_conversion[sensor] + 0.002).round(2)
    return (df['value'].astype(float) * unit_conversion[sensor]).round(2)

def process_sensor_data(data, sensor):
    # Create DataFrame
    df = pd.DataFrame(data)

    # Handle numeric conversion and scaling factor
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['value'].fillna(4, inplace=True)
    df['value'] = apply_scaling_factor(df, sensor)

    # Handle timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort DataFrame by timestamp
    df.sort_values(by='timestamp', inplace=True)

    return df
    
unit_conversion = {
  'Alkalinity (mg/l)': 0.06,
  'Salinity (‰)': 0.00806,
  'DO (mg/l)': 0.2,
  'PH': 1,
  'Temperature (°C)': 1,
  'Turbidity (NTU)': 1
}

sql_command = {
    'PH': f"SELECT sensor, value, timestamp FROM {table_name} WHERE sensor = 'PH' ORDER BY timestamp DESC LIMIT {count}",
    'DO (mg/l)': f"SELECT sensor, value, timestamp FROM {table_name} WHERE sensor = 'DO' ORDER BY timestamp DESC LIMIT {count}",
    'Temperature (°C)': f"SELECT sensor, value, timestamp FROM {table_name} WHERE sensor = 'WT' ORDER BY timestamp DESC LIMIT {count}",
    'Turbidity (V)': f"SELECT sensor, value, timestamp FROM {table_name} WHERE sensor = 'TUR' ORDER BY timestamp DESC LIMIT {count}",
    'Alkalinity (mg/l)': f"SELECT sensor, value, timestamp FROM {table_name} WHERE sensor = 'COND' ORDER BY timestamp DESC LIMIT {count}",
    'Salinity (‰)': f"SELECT sensor, value, timestamp FROM {table_name} WHERE sensor = 'COND' ORDER BY timestamp DESC LIMIT {count}",
}

full_url_ph = f'{api_url}?sql={requests.utils.quote(sql_command["PH"])}'
full_url_do = f'{api_url}?sql={requests.utils.quote(sql_command["DO (mg/l)"])}'
full_url_temp = f'{api_url}?sql={requests.utils.quote(sql_command["Temperature (°C)"])}'
full_url_tur = f'{api_url}?sql={requests.utils.quote(sql_command["Turbidity (V)"])}'
full_url_alk = f'{api_url}?sql={requests.utils.quote(sql_command["Alkalinity (mg/l)"])}'
full_url_sali = f'{api_url}?sql={requests.utils.quote(sql_command["Salinity (‰)"])}'

data = {
    'Dissolved_Oxygen': [],
    'pHs': [],
    'salinities': [],
    'temperatures': [],
    'turbidities': [],
}

df_ph = pd.DataFrame(columns=["sensor", "timestamp", "value"])
df_do = pd.DataFrame(columns=["sensor", "timestamp", "value"])
df_temp = pd.DataFrame(columns=["sensor", "timestamp", "value"])
df_tur = pd.DataFrame(columns=["sensor", "timestamp", "value"])
df_alk = pd.DataFrame(columns=["sensor", "timestamp", "value"])
df_sali = pd.DataFrame(columns=["sensor", "timestamp", "value"])

async def fetch_data(session, url):
    try:
        async with session.get(url, headers={'Content-Type': 'application/json', 'Accept': 'application/json'}) as response:
            return await response.json()
    except aiohttp.ClientConnectorError:
        st.warning("Connection to the server failed. Retrying...")
        await asyncio.sleep(2)  # Add a delay before retrying
        return await fetch_data(session, url)

async def fetch_all_data(api_urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url) for url in api_urls]
        return await asyncio.gather(*tasks)

async def fetch_data_predict(session, api_url, data):
    try:
        async with session.post(api_url) as response:
            data_get = await response.json()

            dissolved_oxygen_predict = data_get['Dissolved_Oxygen_predict']
            phs_predict = data_get['pHs_predict']
            salinities_predict = data_get['salinities_predict']
            temperatures_predict = data_get['temperatures_predict']
            turbidities_predict = data_get['turbidities_predict']

            data['Dissolved_Oxygen'].append(format(float(dissolved_oxygen_predict), '.2f'))
            data['pHs'].append(format(float(phs_predict), '.2f'))
            data['salinities'].append(format(float(salinities_predict), '.2f'))
            data['temperatures'].append(format(float(temperatures_predict), '.2f'))
            data['turbidities'].append(format(float(turbidities_predict), '.2f'))

            

            return data
    except Exception as error:
        print("Error fetching data:", error)
        raise error  # Raise the error to handle it at the calling location

async def fetch_predict_data(data, number_record_predict):
    try:
        async with aiohttp.ClientSession() as session:
            while number_record_predict > 0:
                query_string = "&".join(format_params(key, value) for key, value in data.items())
                api_url = f"https://fad3-115-73-218-247.ngrok-free.app/predict?{query_string}"

                # Use create_task to execute tasks sequentially
                task = asyncio.create_task(fetch_data_predict(session, api_url, data))
                try:
                    data = await task
                except Exception as error:
                    print(f"Error updating data: {error}")
                    # Handle the error, e.g., log it or raise it based on your needs

                number_record_predict -= 1
        return data
    except Exception as error:
        print(f"Error fetching or processing data: {error}")




def format_params(key, values):
    last_12_values = values[-12:]
    return f"{key}={','.join(map(str, last_12_values))}"

def merge_data_predict(df, data, flag, number_record_predict):
        last_timestamp = df["timestamp"].iloc[-1]

        timestamp1 = datetime.fromisoformat(str(df["timestamp"].iloc[-1]))
        timestamp2 = datetime.fromisoformat(str(df["timestamp"].iloc[-12]))
        time_difference = timestamp1 - timestamp2
        time_difference_in_seconds = time_difference.total_seconds()/11
        time_difference_in_seconds = int(time_difference_in_seconds)
        new_rows = pd.DataFrame({
                "sensor": ["PREDICT"] * (number_record_predict + 1),
                "timestamp": [last_timestamp + pd.Timedelta(seconds=time_difference_in_seconds * i) for i in range(number_record_predict + 1)], 
                "value":   data[flag][-(number_record_predict + 1):]
            })
        
        df = pd.concat([df, new_rows])

        return df


def process_data(df, sensor, name_of_chart):
    try:
        if sensor == 'Alkalinity (mg/l)':
            df['sensor'] = df['sensor'].replace('COND', 'ALK')

        if sensor == 'Salinity (‰)':
            df['sensor'] = df['sensor'].replace('COND', 'SALI')

        # Create an area chart using Plotly Express
        fig = px.line(df.head(len(df)), x='timestamp', y='value', line_group='sensor', color='sensor',
                    labels={'value': 'Sensor Value', 'timestamp': 'Timestamp'},
                    title=f'{sensor}')
        
        
        df['value'] = df['value'].astype(float)     
        # Set the Y-axis range based on min and max values
        fig.update_yaxes(range=[df['value'].min() - df['value'].min() * 0.01, df['value'].max() + df['value'].max() * 0.01])

        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        fig.update_layout(height=300)
        # Display the chart using Streamlit
        name_of_chart.plotly_chart(fig, use_container_width=True, theme=None, key=f"{sensor}_chart")

        df = df.reset_index(drop=True)

    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')

async def main():
    ph_chart = st.empty()
    do_chart = st.empty()
    temp_chart = st.empty()
    tur_chart = st.empty()
    sali_chart = st.empty()
    alk_chart = st.empty()
    number_record_predict = 10
    

    api_urls = [
        full_url_ph, full_url_do, full_url_temp, full_url_tur, full_url_alk, full_url_sali
    ]
    while True:
        
        try:
            # Fetch new data in each iteration
            data_ph, data_do, data_temp, data_tur, data_alk, data_sali = await fetch_all_data(api_urls)

            df_ph = process_sensor_data(data_ph, 'PH')
            df_temp = process_sensor_data(data_temp, 'Temperature (°C)')
            df_tur = process_sensor_data(data_tur, 'Turbidity (NTU)')
            df_do = process_sensor_data(data_do, 'DO (mg/l)')
            df_alk = process_sensor_data(data_alk, 'Alkalinity (mg/l)')
            df_sali = process_sensor_data(data_sali, 'Salinity (‰)')

            start_time = time.perf_counter()
            data = {
                'Dissolved_Oxygen': df_do['value'].apply(lambda x: format(x, '.2f')).tolist(),
                'pHs': df_ph['value'].apply(lambda x: format(x, '.2f')).tolist(),
                'salinities': df_sali['value'].apply(lambda x: format(x, '.2f')).tolist(),
                'temperatures': df_temp['value'].apply(lambda x: format(x, '.2f')).tolist(),
                'turbidities': df_tur['value'].apply(lambda x: format(x, '.2f')).tolist(),
            }

            data = await fetch_predict_data(data, number_record_predict)

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            rounded_time = round(elapsed_time, 3)  # Giới hạn 3 chữ số sau dấu thập phân
            print("Time taken to run the function predict with counter 35:", rounded_time, "seconds")

            col2, col3 = st.columns(2)

            with col2:
                df_ph = merge_data_predict(df_ph, data, 'pHs', number_record_predict)
                df_do = merge_data_predict(df_do, data, 'Dissolved_Oxygen', number_record_predict)
                df_temp = merge_data_predict(df_temp, data, 'temperatures', number_record_predict)
                process_data(df_temp, 'Temperature (°C)', temp_chart)
                process_data(df_ph, 'PH', ph_chart)
                process_data(df_do, 'DO (mg/l)', do_chart)

            # Display the second set of area charts and sliders in the second column
            with col3:
                df_sali = merge_data_predict(df_sali, data, 'salinities', number_record_predict)
                df_tur = merge_data_predict(df_tur, data, 'turbidities', number_record_predict)
                process_data(df_tur, 'Turbidity (NTU)', tur_chart)
                process_data(df_sali, 'Salinity (‰)', sali_chart)
                process_data(df_alk, 'Alkalinity (mg/l)', alk_chart)

            # Add a short delay to control the update frequency
            await asyncio.sleep(6)  # Adjust the sleep duration based on your preferred update frequency

        except Exception as e:
            # Handle other exceptions
            st.error(f"An error occurred: {e}")
            await asyncio.sleep(2)  # Add a delay before retrying  
    
# Run the event loop
if __name__ == '__main__':
    asyncio.run(main())



    



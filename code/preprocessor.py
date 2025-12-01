import re
import pandas as pd

def preprocess(data):
    """Preprocess WhatsApp chat data with improved error handling"""
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'

    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    # Validate that we have data
    if not messages or not dates:
        raise ValueError("No valid WhatsApp chat messages found. Please ensure the file is in the correct format.")

    # Ensure messages and dates have the same length
    if len(messages) != len(dates):
        min_len = min(len(messages), len(dates))
        messages = messages[:min_len]
        dates = dates[:min_len]

    df = pd.DataFrame({'user_message': messages, 'date': dates})
    # convert message_date type with error handling
    try:
        df['date'] = pd.to_datetime(df['date'], format=r'%d/%m/%Y, %H:%M - ')
    except Exception as e:
        # Try alternate format if the first one fails
        df['date'] = pd.to_datetime(df['date'], format=r'%m/%d/%Y, %H:%M - ', errors='coerce')


    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('0'))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
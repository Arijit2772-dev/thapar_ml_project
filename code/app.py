import streamlit as st
import preprocessor, helper, ml_models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure matplotlib for dark mode visibility
# Set colors that are bright enough to see in dark mode
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
    'text.color': '#FAFAFA',
    'axes.labelcolor': '#FAFAFA',
    'axes.edgecolor': '#FAFAFA',
    'xtick.color': '#FAFAFA',
    'ytick.color': '#FAFAFA',
    'ytick.labelsize': 10,
    'xtick.labelsize': 10,
})

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        # fetch unique users
        user_list = df['user'].unique().tolist()
        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please make sure you've uploaded a valid WhatsApp chat export file (.txt format)")
        st.stop()

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    # ML Feature Toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("ML Features")
    enable_ml = st.sidebar.checkbox("Enable ML Analysis (Sentiment & Personality)", value=True)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='#00D9FF', linewidth=2)
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='#FF6B6B', linewidth=2)
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='#B19CD9')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='#FFA500')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.heatmap(user_heatmap, cmap='YlOrRd', annot=True, fmt='.0f',
                        linewidths=0.5, cbar_kws={'label': 'Messages'})
        ax.set_xlabel('Time Period', color='#FAFAFA')
        ax.set_ylabel('Day of Week', color='#FAFAFA')
        ax.set_title('Weekly Activity Heatmap', color='#FAFAFA')
        # Make colorbar label visible
        cbar = ax.collections[0].colorbar
        cbar.set_label('Messages', color='#FAFAFA')
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='#FF4B4B')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        ax.axis('off')  # Hide axes for cleaner look
        st.pyplot(fig)

        # most common words
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)


        # Emoji Analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)
            
            with col2:
                # Get top 5 emojis
                top_emojis = emoji_df.head(5)
                
                # Create the pie chart WITHOUT labels
                fig, ax = plt.subplots(figsize=(6, 6))
                
                # Nice color palette
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']
                
                # Create pie chart with only percentages (no labels)
                wedges, texts, autotexts = ax.pie(
                    top_emojis['Count'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    textprops={'fontsize': 14, 'weight': 'bold'}
                )
                
                # Make percentage text white
                for autotext in autotexts:
                    autotext.set_color('white')
                
                # Add title
                ax.set_title('Top 5 Emojis Distribution', fontsize=14, weight='bold', pad=10)
                
                # Display the chart
                st.pyplot(fig)
                
                # Display emoji legend BELOW using Streamlit (this renders emojis properly!)
                st.markdown("#### Legend:")
                for idx, (emoji, count) in enumerate(zip(top_emojis['Emoji'], top_emojis['Count'])):
                    color = colors[idx]
                    st.markdown(
                        f'<span style="color:{color}; font-size:20px;">●</span> '
                        f'<span style="font-size:18px;">{emoji}</span> '
                        f'<span style="color:gray;">({count} times)</span>',
                        unsafe_allow_html=True
                    )
        else:
            st.write("No emojis found in the chat!")

        # ========== MACHINE LEARNING FEATURES ==========
        if enable_ml:
            st.markdown("---")
            st.title("🤖 Machine Learning Analysis")

            # 1. Sentiment Analysis
            st.header("📊 Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                try:
                    sentiment_results = ml_models.sentiment_analysis(df, selected_user)
                except Exception as e:
                    st.error(f"Error in sentiment analysis: {str(e)}")
                    sentiment_results = None

                if sentiment_results and isinstance(sentiment_results, dict) and not sentiment_results.get('df', pd.DataFrame()).empty:
                    col1, col2, col3 = st.columns(3)

                    dist = sentiment_results['distribution']
                    with col1:
                        st.metric("Positive Messages",
                                 dist.get('Positive', 0),
                                 delta=f"{dist.get('Positive', 0)/dist.sum()*100:.1f}%")
                    with col2:
                        st.metric("Neutral Messages",
                                 dist.get('Neutral', 0),
                                 delta=f"{dist.get('Neutral', 0)/dist.sum()*100:.1f}%")
                    with col3:
                        st.metric("Negative Messages",
                                 dist.get('Negative', 0),
                                 delta=f"{dist.get('Negative', 0)/dist.sum()*100:.1f}%")

                    # Sentiment distribution pie chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
                    wedges, texts, autotexts = ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%',
                           colors=colors, startangle=90)
                    # Make text white for visibility
                    for text in texts:
                        text.set_color('#FAFAFA')
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_weight('bold')
                    ax.set_title('Sentiment Distribution', color='#FAFAFA')
                    st.pyplot(fig)

                    # Average polarity
                    st.metric("Average Sentiment Polarity",
                             f"{sentiment_results['avg_polarity']:.3f}",
                             help="Range: -1 (very negative) to +1 (very positive)")

                    # Sentiment timeline
                    if not sentiment_results['timeline'].empty:
                        st.subheader("Sentiment Over Time")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sentiment_results['timeline'].plot(kind='area', stacked=True,
                                                           color=colors, ax=ax, alpha=0.7)
                        ax.set_xlabel('Date', color='#FAFAFA')
                        ax.set_ylabel('Number of Messages', color='#FAFAFA')
                        ax.set_title('Sentiment Timeline', color='#FAFAFA')
                        legend = ax.legend(title='Sentiment')
                        legend.get_title().set_color('#FAFAFA')
                        for text in legend.get_texts():
                            text.set_color('#FAFAFA')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                    # User sentiment comparison (if Overall)
                    if selected_user == 'Overall' and sentiment_results['user_sentiment'] is not None:
                        st.subheader("Most Positive Users")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        user_sent = sentiment_results['user_sentiment']
                        ax.barh(range(len(user_sent)), user_sent.values, color='#00D9FF')
                        ax.set_yticks(range(len(user_sent)))
                        ax.set_yticklabels(user_sent.index)
                        ax.set_xlabel('Average Polarity Score', color='#FAFAFA')
                        ax.set_title('Top 10 Users by Sentiment Positivity', color='#FAFAFA')
                        st.pyplot(fig)
                else:
                    st.info("Not enough data for sentiment analysis")


            # 2. User Personality Insights
            if selected_user != 'Overall':
                st.header("🧠 User Personality Insights")
                try:
                    insights = ml_models.get_user_personality_insights(df, selected_user)
                except Exception as e:
                    st.error(f"Error generating personality insights: {str(e)}")
                    insights = None

                if insights:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Message Style", insights['message_style'])
                        st.metric("Activity Type", insights['activity_type'])
                        st.metric("Emoji Usage", insights['emoji_user'])

                    with col2:
                        st.metric("Avg Message Length", f"{insights['avg_message_length']:.0f} chars")
                        st.metric("Engagement", f"{insights['engagement_percentage']:.1f}%")
                        if insights['avg_response_time_minutes']:
                            st.metric("Median Response Time",
                                     f"{insights['avg_response_time_minutes']:.1f} min")

                    # Personality summary
                    st.subheader("Summary")
                    summary = f"""
                    **{selected_user}** is a **{insights['message_style'].lower()}** communicator who is most active during the **{insights['activity_type'].lower()}**.
                    They contribute **{insights['engagement_percentage']:.1f}%** of the conversation and use emojis at a **{insights['emoji_user'].lower()}** rate
                    ({insights['emoji_per_message']:.2f} per message).
                    """
                    st.info(summary)


                

                     

            



        









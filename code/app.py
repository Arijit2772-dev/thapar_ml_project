import streamlit as st
import preprocessor, helper, ml_models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure matplotlib to work with Streamlit's native theme
plt.rcParams['figure.facecolor'] = 'none'  # Transparent background
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True

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
    enable_ml = st.sidebar.checkbox("Enable ML Analysis", value=True)

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
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = sns.heatmap(user_heatmap, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=0.5)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
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
                    ax.pie(dist.values, labels=dist.index, autopct='%1.1f%%',
                           colors=colors, startangle=90)
                    ax.set_title('Sentiment Distribution')
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
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Number of Messages')
                        ax.set_title('Sentiment Timeline')
                        ax.legend(title='Sentiment')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                    # User sentiment comparison (if Overall)
                    if selected_user == 'Overall' and sentiment_results['user_sentiment'] is not None:
                        st.subheader("Most Positive Users")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        user_sent = sentiment_results['user_sentiment']
                        ax.barh(range(len(user_sent)), user_sent.values, color='skyblue')
                        ax.set_yticks(range(len(user_sent)))
                        ax.set_yticklabels(user_sent.index)
                        ax.set_xlabel('Average Polarity Score')
                        ax.set_title('Top 10 Users by Sentiment Positivity')
                        st.pyplot(fig)
                else:
                    st.info("Not enough data for sentiment analysis")

            # 2. Topic Modeling
            st.header("📚 Topic Discovery")
            col1, col2 = st.columns(2)
            with col1:
                n_topics = st.slider("Number of Topics", 3, 10, 5)
            with col2:
                topic_method = st.selectbox("Method", ['lda', 'nmf'])

            if st.button("Discover Topics"):
                with st.spinner("Running topic modeling..."):
                    try:
                        topics_result = ml_models.topic_modeling(df, selected_user, n_topics, topic_method)
                    except Exception as e:
                        st.error(f"Error in topic modeling: {str(e)}")
                        topics_result = None

                    if topics_result:
                        st.success(f"Discovered {n_topics} topics using {topics_result['method']}")

                        for topic in topics_result['topics']:
                            with st.expander(f"📌 Topic {topic['topic_id']}"):
                                st.write("**Top Keywords:**")
                                st.write(", ".join(topic['top_words']))
                    else:
                        st.warning("Not enough data for topic modeling")

            # 3. Message Clustering
            st.header("🔍 Message Clustering")
            n_clusters = st.slider("Number of Clusters", 3, 8, 5)

            if st.button("Cluster Messages"):
                with st.spinner("Clustering messages..."):
                    try:
                        cluster_result = ml_models.message_clustering(df, selected_user, n_clusters)
                    except Exception as e:
                        st.error(f"Error in message clustering: {str(e)}")
                        cluster_result = None

                    if cluster_result:
                        st.success(f"Messages grouped into {cluster_result['n_clusters']} clusters")

                        # Cluster size distribution
                        cluster_sizes = [c['size'] for c in cluster_result['clusters']]
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(range(len(cluster_sizes)), cluster_sizes, color='teal')
                        ax.set_xlabel('Cluster ID')
                        ax.set_ylabel('Number of Messages')
                        ax.set_title('Cluster Size Distribution')
                        st.pyplot(fig)

                        # Show cluster details
                        for cluster in cluster_result['clusters']:
                            with st.expander(f"Cluster {cluster['cluster_id']} ({cluster['size']} messages)"):
                                st.write("**Representative Message:**")
                                st.info(cluster['representative_message'])
                                st.write("**Sample Messages:**")
                                for i, msg in enumerate(cluster['sample_messages'][:3], 1):
                                    st.text(f"{i}. {msg[:100]}...")
                    else:
                        st.warning("Not enough data for clustering")

            # 4. User Activity Prediction
            if selected_user == 'Overall':
                st.header("🎯 Activity Prediction Model")
                if st.button("Train Prediction Model"):
                    with st.spinner("Training Random Forest model..."):
                        try:
                            prediction_result = ml_models.predict_user_activity(df)
                        except Exception as e:
                            st.error(f"Error in activity prediction: {str(e)}")
                            prediction_result = None

                        if prediction_result:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training Accuracy",
                                         f"{prediction_result['train_accuracy']*100:.2f}%")
                            with col2:
                                st.metric("Test Accuracy",
                                         f"{prediction_result['test_accuracy']*100:.2f}%")

                            st.subheader("Feature Importance")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            feat_imp = prediction_result['feature_importance']
                            ax.barh(feat_imp['feature'], feat_imp['importance'], color='coral')
                            ax.set_xlabel('Importance Score')
                            ax.set_title('Which Features Predict User Activity?')
                            st.pyplot(fig)

                            st.success("Model trained! This predicts which user will message next based on time patterns.")
                        else:
                            st.warning("Not enough data to train prediction model")

            # 5. User Personality Insights
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


                

                     

            



        









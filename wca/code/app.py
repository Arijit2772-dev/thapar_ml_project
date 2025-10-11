import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt


st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    #data_frame ready..

    st.dataframe(df)

    #now would start  analysis...

    #fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall") # for group level analysis..

    selected_user = st.sidebar.selectbox("show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages,words,number_media_messages,num_links = helper.fetch_stats(selected_user,df)

        col1,col2,col3,col4 = st.columns(4)

        with col1:
            st.header("total messages")
            st.title(num_messages)
        
        with col2:
            st.header("words")
            st.title(words)

        with col3:
            st.header("Media Shared")
            st.title(number_media_messages)

        with col4:
            st.header("links Shared")
            st.title(num_links)

        # Finding the busiest users in the group (Group Level ONLY)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = helper.most_busy_users(df)    
            fig,axes = plt.subplots()
            col1,col2 = st.columns(2)

            with col1:
                axes.bar(x.index,x.values,color='Red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        # WordCloud (works for BOTH "Overall" and specific users)
        st.title("WordCloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, axis = plt.subplots()
        axis.imshow(df_wc)
        plt.axis('off')  # Remove axes for cleaner look
        st.pyplot(fig)

        most_common_df = helper.most_common_words(selected_user,df)

        fig,axis = plt.subplots()
        plt.xticks(rotation = 'vertical')
        
        axis.barh(most_common_df[0],most_common_df[1])


        st.title('Most common Words')
        st.pyplot(fig)



                

                     

            



        









import streamlit as st 

# 야후금융에서 주식정보를 제공하는 라이브러리 yfiance 이용해서
# 주식정보를 불러오고 차트 그린다.

# 해당 주식에 대한 트윗글들을 불러올 수 있는 API 가 있다 
# stocktwits.com 에서 제공하는 Restful API 를 호출해서 
# 데이터 가져오는것 실습

# 이것이 되면 네이버, 카카오에서도 api 가져와서 사용가능 하다 

import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd 
import numpy as np
# api  호출을 위한 라이브러리 임폴트
import requests
# 프로펫 라이브러리 임폴트
from fbprophet import Prophet


def main() :
    st.header('Online Stock Price Ticker')

    # yfiance 실행
 
    symbol = st.text_input('심볼 입력 : ')  # 이렇게 하면 밑에 심볼을 포맷으로 해놓은것에 유저가 입력할때마다 들어가서 보고싶은 회사의 주식을 볼수 있음
     # symbol = 'MSFT'  # MSFT 는 마이크로 소프트   # 내가 보고싶은 종목(회사)를 바꾸고 싶으면 여기를 바꾸면 됨  
    data =  yf.Ticker(symbol) 

    today = datetime.now().date().isoformat()  # 연월일 만 가져올땐 date(), 시분초만 가져올땐 time()  # isoformat는 숫자들을 문자열로 
    print(today)  

    df = data.history(start= '2010-06-01', end= '2021-03-22')
    st.dataframe(df)
 
    st.subheader('종가')  # 종가 차트 그리기
    st.line_chart(df['Close'])

    st.subheader('거래량')   # 거래량 차트 그리기
    st.line_chart(df['Volume'])

    
    
    # yfiance 라이브러리만의 정보
    # data.info  # 내가 보는 회사의 정보들을 알려줌
    # data.calendar  
    # data.major_holders # 대주주가 누구인가를 알수 있음
    # data.institutional_holders  # 어떤사람들이 주식 가지고 있는지 알수 있음
    # data.recommendations  # 어느기관이 샀는지 시간별로 알수 있음 
    # data.dividends # 배당금 정보 알 수 있음

    div_df = data.dividends
    st.dataframe( div_df.resample('Y').sum() )  # 배당금 각 연도별로 알수있다  # resample !!
    
    new_df = div_df.reset_index()  # 지금 현재 날짜를 인덱스 번호로 해서 컬럼으로 만든다   # 그래서 연도별 배당금을 알수있다 차트로 # 이렇게 바꾸면 이따가 prophet에도 쉽게 사용가능 하다 
    new_df['Year'] = new_df['Date'].dt.year 
    # 배당금을 차트로
    fig =  plt.figure()
    plt.bar(new_df['Year'], new_df['Dividends'])
    st.pyplot(fig)



    # 여러 주식 데이터를 한번에 보여주기 
    favorites = ['msft', 'tsla','nvda','aapl','amzn']  # 이거를 심볼로 놔도됨
    f_df = pd.DataFrame()  # 각각의 종가(Close)들을 비어있는 데이터프레임에 넣어서 보여주고 싶을 때
    for stock in favorites :
        f_df[stock] = yf.Ticker(stock).history(start='2010-01-01',end=today)['Close']  # 변수로 저장해서 2-3줄로도 가능 
        # 비어있던 f_df에 stock라고 만든 컬럼에 이값들을 넣어달라
    st.dataframe(f_df)
    # 위에것을 한번에 차트 나타내라
    st.line_chart(f_df)   # 이런부분 멀티셀렉트로 하면 좋을 것 같다 



    # API 호출을 위한 라이브러리 임폴트 하고 와야해
    # 구글에 주소 가져올때 했었음
    # pip install requests 이거 설치하면 이제 불러올수 있고 stocktwits 주소 넣으면 됨
    res = requests.get('https://api.stocktwits.com/api/2/streams/symbol/{}.json'.format(symbol)) # 원하는 회사의 데이터를 보고 싶을 때 # 이렇게 만들었으면 맨 위에 text_input 만들어서 보게 할수도 있음 
    # json 형식이므로 , .json()
    res_data =  res.json()  # 리스트와 딕셔너리 형태로 불러옴
    # 파이썬의 딕셔너리와 리스트이 조합으로 사용가능  
    #st.write(res_data)

    for message in  res_data['messages'] :
         # message 키에 있는 내용들 가져와서
        #st.write(message)  # 메세지 30개의 전체 내용들을 볼수 있음

        col1, col2 = st.beta_columns( [1,4] )  # 1 : 4의 비율로 보여달라

        with col1 :
            st.image( message['user']['avatar_url'] )
        with col2 :
            st.write('유저 이름 : ' + message['user']['username'] ) 
            st.write('트윗 내용 : ' + message['body'] )       
            st.write( '올린 시간 : ' + message['created_at'] )


        # st.image( message['user']['avatar_url'] )   # col, with 빼고 포문에 이것만 넣으면 다 한줄에 그냥 볼수있고  위에처럼 바꾸면 예쁘게 정리한모습으로 볼수 있음
        # st.write( message['user']['username'] ) 
        # st.write( message['body'] )       
        # st.write( message['created_at'] )



    # 여기서 부터는 프로펫 !!!
    p_df = df.reset_index()  
    p_df.rename( columns = { 'Data' : 'ds', 'Close' : 'y'}, inplace = True )
    # st.dataframe(p_df)

    # 이제 예측가능 !
    m = Prophet()
    m.fit(p_df)
    future = m.make_future_dataframe(periods= 365)  # 1년치 데이터를 만들어 놓고  
    forecast =  m.predict(future)  #  >>> 예측해달라
    st.dataframe(forecast)

    # 위에 차트 그려라
    fig1 =  m.plot(forecast)
    st.pyplot(fig1)

    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)  

    # 이렇게 했는데 message api를 많이 써서 볼수가없다고 뜬다  그래서 res = requests.get~~ 부터 프로펫 전까지 주석 처리하고 하면 보인다

    # 이제 파파고 api  같은 것들 가져와서 쓸수 있다 


if __name__ == '__main__' :
    main()
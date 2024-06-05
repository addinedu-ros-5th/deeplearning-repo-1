import mysql.connector
import streamlit as st

def connection():
    conn = mysql.connector.connect(host=st.secrets.db.host,
                                   user=st.secrets.db.user,
                                   port=st.secrets.db.port,
                                   password=st.secrets.db.password,
                                   database=st.secrets.db.database)
    return conn

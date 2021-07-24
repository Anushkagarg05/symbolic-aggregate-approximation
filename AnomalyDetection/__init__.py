import datetime
import logging
import pandas as pd
import os
from io import StringIO
import AnomalyDetector
import logging

if __name__=="__main__":
    res=""
    df = pandas.read_csv("../Data/NSESAMPLESTOCKS.csv",sep=',')
    logging.info('the size of dataframe is : %s',df.shape[0])


    keyField = os.environ["Key"]
    dateField = os.environ["Date"]
    valueField = os.environ["Value"]

    anomaly_detector = AnomalyDetector.SAX(df, 
                                            key_field_name = keyField, 
                                            date_field_name = dateField, 
                                            value_field_name = valueField, 
                                            window_length = 4, 
                                            select_fields = [keyField],
                                            partial_latest_snapshot = False)

    counter = 0
    anomaly_records = "Stock\tDate\tRank\tValue\n"
    for detected_discords in anomaly_detector:
        for detected_discord in detected_discords:
                    counter=counter+1
                    anomaly_record = '\t'.join([str(field) for field in detected_discord]) + '\n'
                    anomaly_records = anomaly_records + anomaly_record
        logging.info('Counter : %d',counter)
        res = res+anomaly_records

    df_result = pd.read_csv(StringIO(res), sep="\t")
    df_result.to_csv("results.csv",sep='\t',index=False)
    logging.info('SUCCESSFUL UPLOAD')

import numpy as np
import pandas as pd
import os
import ipaddress

from sklearn.model_selection import train_test_split

mac2int = lambda mac: int(mac.translate(str.maketrans('', '', ":.- ")), 16)


def try_convert_mac(value) -> int:
    """
    Tries to convert MAC to integer. If failure, returns NaN
    :param value: MAC address
    :return: integer equivalent of value / np.nan
    """
    try:
        return mac2int(value)
    except:
        return np.nan

def try_convert_ip(value) -> int:
    """
    Tries to convert IP to integer. If failure, returns NaN
    :param value: IP address
    :return: integer equivalent of value / np.nan
    """
    try:
        return ipaddress.ip_address(value)._ip
    except (ValueError):
        """Not a valid IP address..."""
        return np.nan

def try_convert_protocol(value) -> int:
    """
    Tries to convert numeric Protocol to integer. If failure, returns NaN
    :param value: protocol value
    :return: integer equivalent of value / np.nan
    """
    try:
        return int(value)
    except:
        if value == "rarp": return 32821
        else:
            return np.nan

def try_convert_port(value) -> int:
    """
    Tries to convert numeric port to integer. If failure, returns NaN
    :param value: protocol value
    :return: integer equivalent of value / np.nan
    """
    try:
        return int(value)
    except Exception:
        try:
            return int(value, 16)
        except:
            return np.nan

def encode(dataset: pd.DataFrame, feature_data) -> pd.DataFrame:
    """
    Encode given columns to be used in a Neural Network transforming all values to integers
    :param dataset: (DataFrame) pandas Dataframe to transform
    :param feature_data: list of feature:type pairs
    :return: (DataFrame) encoded dataset
    """
    dataset = dataset.fillna(0)

    encoded_dataset = pd.DataFrame(index=dataset.index)

    for item in feature_data:
        col = item["feature"]
        type = item["type"]

        "Handle special types like ip or mac addresses"""
        if type == 'ip':
            data_series = dataset[col].apply(try_convert_ip)
        elif type == 'mac':
            data_series = dataset.loc[:, col].apply(try_convert_mac)
        elif type == 'port':
            data_series = dataset.loc[:, col].apply(try_convert_port)
        elif type == 'protocol':
            data_series = dataset.loc[:, col].apply(try_convert_protocol)
        else:
            "Do not transform"
            data_series = dataset.loc[:, col]

        encoded_dataset[col] = pd.DataFrame(data_series, index=dataset.index)

    encoded_dataset = encoded_dataset.dropna()
    """Convert other columns to numeric"""
    return encoded_dataset.apply(pd.to_numeric)


if __name__ == "__main__":

    normal_dir = "./normal/"
    attack_dir = "./attack/"

    data_normal = pd.DataFrame()

    cur_dir = os.path.dirname(__file__)

    for file in os.listdir(os.path.join(cur_dir, normal_dir)): 
        normal = pd.read_csv(os.path.normpath(os.path.join(cur_dir, normal_dir, file)))
        data_normal = pd.concat([data_normal, normal])

    data_attack = pd.DataFrame()
    for file in os.listdir(os.path.join(cur_dir, attack_dir)): 
        attack = pd.read_csv(os.path.normpath(os.path.join(cur_dir, attack_dir, file)))
        data_attack = pd.concat([data_attack, attack])

    print(data_normal.shape)
    print(data_attack.shape)

    """ Preprocessing """

    feature_list = ["saddr", "smac", "daddr", "dmac","proto","dport","spkts",
                    "dpkts","pkts","sbytes","dbytes","bytes","dur","sintpkt","dintpkt"]

    feature_data = [
      {
          "feature": "saddr",
          "type": "ip"
      },
      {
          "feature": "smac",
          "type": "mac"
      },
      {
          "feature": "daddr",
          "type": "ip"
      },
      {
          "feature": "dmac",
          "type": "mac"
      },
      {
          "feature": "proto",
          "type": "protocol"
      },
      {
          "feature": "dport",
          "type": "port"
      },
      {
          "feature": "spkts",
          "type": "int"
      },
      {
          "feature": "dpkts",
          "type": "int"
      },
      {
          "feature": "pkts",
          "type": "int"
      },
      {
          "feature": "sbytes",
          "type": "int"
      },
      {
          "feature": "dbytes",
          "type": "int"
      },
      {
          "feature": "bytes",
          "type": "int"
      },
      {
          "feature": "dur",
          "type": "float"
      },
      {
          "feature": "sintpkt",
          "type": "float"
      },
      {
          "feature": "dintpkt",
          "type": "float"
      }
    ]

    data_normal = data_normal[feature_list]
    data_attack = data_attack[feature_list]

    print("NORMAL")
    print(data_normal.describe())
    
    print("ATTACK")
    print(data_attack.describe())

    normal_encoded = encode(data_normal, feature_data)
    attack_encoded = encode(data_attack, feature_data)

    

    normal_encoded["label"] = 0
    attack_encoded["label"] = 1

    all = pd.concat([normal_encoded, attack_encoded])
    X = all.loc[:, all.columns != 'label']
    y = all["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True).squeeze()
    y_test = y_test.reset_index(drop=True).squeeze()

    X_train.to_csv(os.path.join(cur_dir, "X_train.csv"))
    X_test.to_csv(os.path.join(cur_dir, "X_test.csv"))
    y_train.to_csv(os.path.join(cur_dir, "y_train.csv"))
    y_test.to_csv(os.path.join(cur_dir, "y_test.csv"))

    data_normal.describe().to_csv(os.path.join(cur_dir, "normal_desc.csv"))
    data_attack.describe().to_csv(os.path.join(cur_dir, "attack_desc.csv"))
    
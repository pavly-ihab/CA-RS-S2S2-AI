# config/config.py

# Path to dataset
DATA_PATH = "C:\\Users\\bavly\\OneDrive\\Desktop\\depi_cairo_AI\\CA-RS-S2S2-AI\\src\\machine learning\\session 8\\assignment\\project titanic\\data\\raw\\Titanic.csv"

# Columns to drop from the dataset
# Adjust these based on what features are unnecessary for your model
COLS_TO_DROP = [
    "PassengerId", 
    "Name", 
    "Ticket", 
    "Cabin"
]

# Categorical columns to convert
CAT_COLS = [
    "Survived", 
    "Pclass", 
    "Sex", 
    "SibSp", 
    "Parch", 
    "Embarked"
]
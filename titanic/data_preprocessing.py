import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop PassengerId.
train.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)

# Whether it has Cabin number
train['Has_Cabin'] = train.Cabin.apply(lambda x: 0 if pd.isna(x) else 1)
test['Has_Cabin'] = test.Cabin.apply(lambda x: 0 if pd.isna(x) else 1)

# Drop Cabin because it is almost Nan or null.
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# Add FamilySize
train['FamilySize'] = train.SibSp + train.Parch + 1
test['FamilySize'] = test.SibSp + test.Parch + 1

# Add isAlone
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
test['IsAlone'] = 0
test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

# Drop SibSp
train.drop('SibSp', axis=1, inplace=True)
test.drop('SibSp', axis=1, inplace=True)


# Change to median if Age and Fare(test only) is Nan or null.
def set_median(target, keys):
    for key in keys:
        age_median = target[key].median()
        target[key] = target[key].fillna(age_median)
    return target
train = set_median(train, ['Age'])
test = set_median(test, ['Age', 'Fare'])


# Categorize Age
def categorize_age(x):
    if x <= 15:
        return 0
    if 15 < x and x <= 30:
        return 1
    if 30 < x and x <= 45:
        return 2
    if 45 < x and x <= 60:
        return 3
    return 4
train.Age = train.Age.apply(categorize_age)
test.Age = train.Age.apply(categorize_age)


# Categorize Fare
def categorize_fare(x):
    if x <= 7:
        return 0
    if 7 < x and x <= 14:
        return 1
    if 7 < x and x <= 31:
        return 2
    return 3
train.Fare = train.Fare.apply(categorize_fare)
test.Fare = train.Fare.apply(categorize_fare)


# Change Sex to integer.
def replace_sex(target):
    target.Sex = target.Sex.replace('male', 0)
    target.Sex = target.Sex.replace('female', 1)
    return target
train = replace_sex(train)
test = replace_sex(test)

# Drop Ticket because it is roughly understood by Pclass.
train.drop('Ticket', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)

'''
Change to the most port if it is Nan or null.
S    644
C    168
Q     77
'''
train.Embarked.fillna('S', inplace=True)


# Categorize Embarked with one hot encoding.
def encode_one_hot_embarked(target):
    embarked_encoded, embarked_categories = target.Embarked.factorize()
    encoder = OneHotEncoder()
    embarked_hot = encoder.fit_transform(embarked_encoded.reshape(-1, 1))
    embarked_hot.toarray()
    df_embarked = pd.DataFrame(
        embarked_hot.toarray(),
        columns=embarked_categories
    )
    return pd.concat(
        [target.drop('Embarked', axis=1), df_embarked],
        axis=1,
        join='inner'
    )
train = encode_one_hot_embarked(train)
test = encode_one_hot_embarked(test)


# Categorize Name to honorific.
def encode_one_hot_name(target):
    categories = ['Mr.', 'Mis.', 'Miss.', 'Mrs.', 'Master']
    data = []
    for i, v in target.Name.iteritems():
        for c in categories:
            if c in v:
                data.append(c)
                break
        else:
            if train.Sex[i] == 0:
                data.append('Mr.')
            else:
                data.append('Ms.')

    name_encoded, name_categories = pd.Series(data).factorize()
    encoder = OneHotEncoder()
    name_hot = encoder.fit_transform(name_encoded.reshape(-1, 1))
    name_hot.toarray()
    df_name = pd.DataFrame(name_hot.toarray(), columns=name_categories)
    return pd.concat(
        [target.drop('Name', axis=1), df_name],
        axis=1,
        join='inner'
    )
train = encode_one_hot_name(train)
test = encode_one_hot_name(test)

print('/_/_/_/_/_/_/')
print('/_/ train /_/')
print('/_/_/_/_/_/_/')
print(train.head(10))
print('/_/_/_/_/_/_/')
print('/_/ test /_/')
print('/_/_/_/_/_/')
print(test.head(10))

train.to_csv('./preprocessed_train.csv')
test.to_csv('./preprocessed_test.csv')

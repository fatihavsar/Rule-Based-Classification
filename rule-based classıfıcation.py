


#############################################
# Calculation of potential customer earnings with rule-based classıfıcation
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# Business Problem
#############################################
#A game company wants to create level-based new customer definitions by using some features of its customers
#and to create segments according to these new customer definitions and estimates what would be the average income that
#new customers will bring to the company according to these segments.

## For example: It is desired to determine how much a 25-year-old male user from Turkey, who is an IOS user, can earn on average.

#############################################
# Dataset Story
#############################################
#Persona.csv dataset contains the prices of the products sold by an international game company and some
#demographic information of the users who buy these products. The data set consists of records created in each sales transaction. This means that the table is not deduplicated.
#In other words, a user with certain demographic characteristics may have made more than one purchase.


# Price: Amount of the customer's expense
# Source: Type of the device that customers connect
# Sex: Customer's gender
# Country: Customer's country
# Age: Customer's age

################# Before Classification #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After Classification #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJECT TASKS
#############################################

#############################################
# TASK 1: Answer the following questions.
#############################################

# Question 1: Read the persona.csv file and show the general information about the dataset.


import pandas as pd
pd.set_option("display.max_rows", None)
df=pd.read_csv(r"C:\Users\User\OneDrive\Masaüstü\Kural Tabanlı Sınıflandırma/persona.csv")
df.head()
df.tail()
df.shape
df.describe()
df.value_counts()
df.info


# # Question 2: How many unique SOURCE are there? What are their frequencies?

df.nunique()
df["SOURCE"].nunique()
df.value_counts("SOURCE")


# Question 3: How many unique PRICEs are there?

df["PRICE"].nunique()


# Question 4: How many sales were made from which PRICE?

df["PRICE"].value_counts()


# Question 5: How many sales were made from which country?
df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()


# Question 6: How much was earned in total from sales by country?

df.groupby("COUNTRY")["PRICE"].sum()

df.groupby("COUNTRY").agg({"PRICE":"sum"})

# Question 7: What are the sales numbers by SOURCE types?

df["SOURCE"].value_counts()

df.groupby("SOURCE").agg({"PRICE":"sum"})
df.groupby("SOURCE")["PRICE"].sum()

# Question 8: What are the PRICE averages by country?df.groupby("COUNTRY")["PRICE"].mean()
df.groupby(by=["COUNTRY"]).agg({"PRICE":"mean"})
df.groupby(by=['COUNTRY']).agg({"PRICE": "mean"})


# Question 9: What are the PRICE averages by SOURCEs?

df.groupby("SOURCE")["PRICE"].mean()

# Question 10 :What are the PRICE averages in the COUNTRY-SOURCE diffraction?
df.groupby(["COUNTRY","SOURCE"])["PRICE"].mean()

df.groupby(by=(["COUNTRY","SOURCE"])).agg({"PRICE": "mean"})

#############################################
# TASK 2: What are the average earnings in diffraction of COUNTRY, SOURCE, SEX, AGE?
#############################################
df.groupby(by=["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).head()

#############################################
# TASK 3: Sort the output by PRICE
#############################################
# Apply the sort_values method to PRICE in descending order to see the output in the previous question better.
# Save the output as agg_df.

agg_df = df.groupby(by=["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()


#############################################
## TASK 4: Convert the names in the index to variable names.
#############################################
# All variables except PRICE in the output of the third question are index names.
# Convert these names to variable names.

agg_df = agg_df.reset_index()
agg_df.head()


#############################################
# TASK 5: Convert AGE variable to categorical variable and add it to agg_df.
#############################################
# Convert the numeric variable age to a categorical variable.
# Construct the intervals in a way that you think will be persuasive.
# For example: '0_18', '19_23', '24_30', '31_40', '41_70'

bins = [0,18,23,30,40,agg_df["AGE"].max()]

mylabels= ['0_18','19_23','24_30','31_40','41_'+ str(agg_df["AGE"].max())]

agg_df["age_cat"]=pd.cut(agg_df["AGE"],bins,labels=mylabels)
agg_df.head()

#####
ücret=[0,10,20,30, agg_df["PRICE"].max()]
ücretaralık=['ucuz','orta','pahalı','çok'+ str(agg_df["PRICE"].max())]
agg_df["price_cat"]=pd.cut(agg_df["PRICE"],ücret,labels=ücretaralık)
agg_df.tail() #####
#####


#############################################
# TASK 6: Identify new level based customers and add them as variables to the dataset.
#############################################
# Define a variable named customers_level_based and add this variable to the dataset.
# After creating customers_level_based values with list comp, these values need to be deduplicated.
# For example, it could be more than one of the following expressions: USA_ANDROID_MALE_0_18
#It is necessary to take them by groupby and get the price averages.



agg_df["customers_level_based"]=agg_df[["COUNTRY","SOURCE","SEX","age_cat"]].agg(lambda x : "_".join(x).upper(),axis=1)
agg_df.head()

agg_df["customers_level_based"].value_counts()

agg_df=agg_df.groupby("customers_level_based").agg({"PRICE":"mean"})

agg_df=agg_df.reset_index("customers_level_based")
agg_df.head()

agg_df["customers_level_based"].value_counts()
agg_df.head()


#############################################
# TASK 7: Segment new customers (USA_ANDROID_MALE_0_18)
#############################################
# Segment by PRICE,
# add segments to agg_df with the naming "SEGMENT",
# describe the segments,

agg_df["SEGMENT"]=pd.qcut(agg_df["PRICE"],4,labels=["D","C","B","A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE":"mean"})


#############################################
# TASK 8: Classify the new customers and estimate how much income they can bring.
#############################################
# Which segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is expected to earn on average?

new_customer="TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"]==new_customer]

# In which segment and on average how much income would a 35-year-old French woman using iOS expect to earn?

new_customer2="FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"]==new_customer2]
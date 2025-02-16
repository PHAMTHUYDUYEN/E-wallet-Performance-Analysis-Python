# [Python] E-wallet Performance Analysis 

## I. INTRODUCTION
In this project, Python techniques: **_Pandas, NumPy, EDA and data wrangling methods_** (cleaning, aggregation, and feature engineering) were utilized to analyze **_payment transactions, product performance, and team contributions_**. This project aimed to gain insights for decision-making enhancement in payment operations, product strategy, and financial reporting.

## II. DATASETS
Datasets used (as attachment) is of an e-wallet company. There are 3 datasets with description as below table:
| No. | Dataset             | Description                          |
|----|---------------------|----------------------------------|
| 1  | payment_report.csv | Monthly payment volume of products |
| 2  | product.csv        | Product information               |
| 3  | transactions.csv   | Transactions information         |

## III. KEY AREAS TO ANALYZE
To gain insights from given datasets, the analysis focus on 3 key areas, which aim to help decision-making process:

**1. Product Sales Performance**
- `Top 3 products` with highest sales volume.
- `Data integrity`: each product belongs to a single team?
- `Lowest-performing team` since Q2 2023. 
- `Least-contributing category`.
  
**2. Refund Transaction Analysis**
- `Distribution of refund` transactions by source.
- Identifying `highest refund-contributing source`.
  
**3. Transaction Classification & Performance Metrics**
- `Categorizing transactions` based on predefined business rules.
- Calculating `transaction statistics` for each transaction type.

## IV. DATA PREPARATION
**1. Checking**
```Python
# Checking for Missing Values
print("---",payment_report.isna().sum())
print("---",product.isna().sum())
print("---",transactions.isna().sum())

# Checking for Duplications
print(payment_report.shape)
print(payment_report.nunique())
print(product.shape)
print(product.nunique())
print(transactions.shape)
print(transactions.nunique())

# Checking data type
print(payment_report.info())
print(product.info())
print(transactions.info())

# Checking data value
print(payment_report.describe())
print(product.describe())
print(transactions.describe())
```

**2. Conclusion & Handling**
- Missing data:
  - In Transactions table:
    42183 rows in sender_id -> No base to fill in -> No action
    138933 rows in receiver_id -> No base to fill in -> No action
    1130449 rows in extra_info -> No action

- Duplicates:
  - In PK transaction_id of Transactions table:
    28 duplicated rows -> Delete rows

- Incorrect data types:
  - All data types are correct

- Incorrect values:
  - In Transactions table:
    receiver_id has negative values -> Convert all to positive
    transStatus has negative values -> Don’t understand meaning -> No action

```Python
# Drop duplicates in Transactions df
transactions.drop_duplicates(inplace = True)

# Convert receiver_id to positive
transactions["receiver_id"] = np.abs(transactions["receiver_id"])
```
## V. DATA WRANGLING
### **Problem #1: Find top 3 product_ids with the highest volume.**
```Python
# Merge payment_report df with product df
payment_enriched = pd.merge(payment_report, product, on="product_id", how="outer")

# Find top 3 products
volume_by_product = payment_enriched.groupby("product_id")["volume"].sum()
volume_by_product.nlargest(3)
```
| product_id | volume        |
|------------|--------------|
| **1976**   | 6.179758e+10 |
| **429**    | 1.466768e+10 |
| **372**    | 1.371366e+10 |

:arrow_right: Top 3 products are ones having id: 1976, 429, 372.

---
### **Problem #2: Given that 1 product_id is only owed by 1 team, are there any abnormal products against this rule?**
```Python
team_by_product = payment_enriched.groupby("product_id").agg({"team_own":"count"})
team_by_product[team_by_product["team_own"] != 1]
```
| product_id | team_own |
|------------|---------|
| 3          | 0       |
| 12         | 4       |
| 15         | 2       |
| 17         | 4       |
| 18         | 4       |
| ...        | ...     |
| 2356       | 3       |
| 2587       | 2       |
| 10033      | 0       |
| 10039      | 4       |
| 15067      | 3       |

:arrow_right: Above is a list of all abnormal products, which can be used for further correction activities.

---
### **Problem #3: Find the team has had the lowest performance (lowest volume) since Q2.2023. Find the category that contributes the least to that team.**

_1. Lowest-performing team:_
```Python
# Filter values in Q2.2023
q2_2023 = payment_enriched[payment_enriched["report_month"].isin(["2023-04","2023-05","2023-06"])]

# Calculate volume by team
volume_by_team = q2_2023.groupby("team_own").agg({"volume":"sum"})

# Identify lowest performance team
volume_by_team["volume"].nsmallest(1)
```
| team_own | volume     |
|----------|-----------|
| **APS**  | 51141753.0 |

_2. Least-contributing category of such team:_
```Python
# Filter values of APS team
aps = payment_enriched[payment_enriched["team_own"] == "APS"]

# Calculate volume by category
volume_by_category = aps.groupby("category").agg({"volume":"sum"})

# Identify lowest contributed category
volume_by_category["volume"].nsmallest(1)
```
| category  | volume |
|-----------|--------|
| **PXXXXXO** | 0.0    |

➡️ Team with lowest performance since Q2.2023 was APS with PXXXXXO being the lowest contributed category.

---
### **Problem #4: Find the contribution of source_ids of refund transactions (payment_group = ‘refund’), what is the source_id with the highest contribution?**
```Python
# Filter 'refund' payment group
refund = payment_enriched[payment_enriched["payment_group"] == "refund"]

# Calculate volume by source_id
volume_by_souce = refund.groupby("source_id").agg({"volume":"sum"})

# Identify source_id having highest volume
volume_by_souce["volume"].nlargest(1)
```
| source_id | volume        |
|-----------|--------------|
| **38.0**  | 3.652745e+10 |

➡️Source number #38 contributed most refund transactions. This source should be further examined to find high-refund causes.

---
### **Problem #5: Categorize type of transactions (‘transaction_type’) for each row, given:**
- transType = 2 & merchant_id = 1205: Bank Transfer Transaction
- transType = 2 & merchant_id = 2260: Withdraw Money Transaction
- transType = 2 & merchant_id = 2270: Top Up Money Transaction
- transType = 2 & others merchant_id: Payment Transaction
- transType = 8, merchant_id = 2250: Transfer Money Transaction
- transType = 8 & others merchant_id: Split Bill Transaction
- Remained cases are invalid transactions

```Python
# Create list of Transaction type
transaction_category = ["Bank Transfer Transaction", "Withdraw Money Transaction", "Top Up Money Transaction", "Payment Transaction", "Transfer Money Transaction", "Split Bill Transaction"]

# Create list of Conditions
conditions = [
              ((transactions["transType"]==2)&(transactions["merchant_id"]==1205)),
              ((transactions["transType"]==2)&(transactions["merchant_id"]==2260)),
              ((transactions["transType"]==2)&(transactions["merchant_id"]==2270)),
              (transactions["transType"]==2),
              ((transactions["transType"]==8)&(transactions["merchant_id"]==2250)),
              (transactions["transType"]==8)
              ]

# Create new column 'transaction_type'
transactions["transaction_type"] = np.select(conditions, transaction_category, default="Invalid Transaction")

print(transactions.head())
```
| transaction_id | merchant_id | volume  | transType | transStatus | sender_id  | receiver_id | extra_info | timeStamp      | transaction_type         |
|---------------|------------|---------|-----------|-------------|------------|-------------|------------|----------------|-------------------------|
| 3002692434    | 5          | 100000  | 24        | 1           | 10199794.0 | 199794.0    | NaN        | 1682932054455  | Invalid Transaction     |
| 3002692437    | 305        | 20000   | 2         | 1           | 14022211.0 | 14022211.0  | NaN        | 1682932054912  | Payment Transaction     |
| 3001960110    | 7255       | 48605   | 22        | 1           | NaN        | 10530940.0  | NaN        | 1682932055000  | Invalid Transaction     |
| 3002680710    | 2270       | 150000  | 2         | 1           | 10059206.0 | 59206.0     | NaN        | 1682932055622  | Top Up Money Transaction |
| 3002680713    | 2275       | 90000   | 2         | 1           | 10004711.0 | 4711.0      | NaN        | 1682932056197  | Payment Transaction     |

➡️ All transactions are categorized, which is helpful for deeper analysis.

---
### **Problem #6: Of each transaction type (excluding invalid transactions): find the number of transactions, volume, senders and receivers.**
```Python
# Count transaction, volume, senders, receivers of each transaction type
grouped_rs_transaction = transactions.groupby("transaction_type").agg(
                                    transaction_count=("transaction_id","count"),
                                    volume_count=("volume","count"),
                                    sender_count=("sender_id","count"),
                                    receiver_count=("receiver_id","count")
                                  ).reset_index()

# Exclude invalid transactions
grouped_rs_transaction[grouped_rs_transaction["transaction_type"] != "Invalid Transaction"].reset_index(drop=True)
```
| transaction_type              | transaction_count | volume_count | sender_count | receiver_count |
|------------------------------|------------------|-------------|-------------|---------------|
| Bank Transfer Transaction     | 37,879          | 37,879      | 37,879      | 14,004        |
| Payment Transaction          | 398,665         | 398,665     | 398,665     | 259,982       |
| Split Bill Transaction       | 1,376           | 1,376       | 1,376       | 1,376         |
| Top Up Money Transaction     | 290,498         | 290,498     | 290,498     | 290,498       |
| Transfer Money Transaction   | 341,173         | 341,173     | 341,173     | 341,173       |
| Withdraw Money Transaction   | 33,725          | 33,725      | 33,725      | 33,725        |

➡️ What can be seen?
- **Payment transactions dominate** with 398,665 transactions and 259,982 receivers --> **Core** focus for **optimization**.
- **Bank transfers** have 37,879 transactions but only 14,004 receivers --> **Bulk payments** or **business payouts**.
- **Top-up** transactions are **highly frequent** at 290,498 --> Opportunity to **incentivize higher deposits**.
- **Split bill** transactions have only 1,376 transactions --> **Require better promotion** or integration with social payments.

## VI. CONCLUSION
In conclusion, my analysis of the e-Wallet performance using Python has uncovered valuable **insights** into **payment transactions, product performance, and team contributions,** which can inform future business decisions. The **next step** will involve **visualizing** or **further analysis** into some low performance aspects. Overall, this project demonstrates the effectiveness of employing Python to derive meaningful insights from multiple datasets.

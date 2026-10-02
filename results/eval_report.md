# Evaluation: vector vs hybrid search (20 queries, k=5)

| Metric | Vector | Hybrid |
|---|---|---|
| constraint_precision | 0.81 | 1.00 |
| full_precision | 0.59 | 0.73 |
| recall | 0.70 | 0.89 |

Query parser accuracy (exact filter match): 100%

## Per query (full_precision)

| Query | Relevant | Vector | Hybrid | Parser OK |
|---|---|---|---|---|
| 4 bedroom house with a fireplace under $800k | 5 | 0.6 | 0.8 | yes |
| condo under $500k | 20 | 1.0 | 1.0 | yes |
| 3 bedroom townhouse | 16 | 0.8 | 1.0 | yes |
| a home with a pool | 16 | 1.0 | 1.0 | yes |
| house with at least 5 bedrooms | 19 | 1.0 | 1.0 | yes |
| cheap starter home under $350k | 42 | 0.8 | 1.0 | yes |
| luxury home over $2 million with a pool | 2 | 0.2 | 0.4 | yes |
| 2 bedroom condo with a balcony | 6 | 0.6 | 0.8 | yes |
| house with a garage under $600k | 6 | 0.4 | 0.4 | yes |
| at least 2 bathrooms and a home office | 8 | 0.8 | 0.8 | yes |
| small place under 1000 square feet | 17 | 0.6 | 1.0 | yes |
| house over 3000 sqft with mountain views | 2 | 0.0 | 0.4 | yes |
| townhouse under $450k | 20 | 0.6 | 1.0 | yes |
| home with water views | 3 | 0.4 | 0.4 | yes |
| 3 bedroom house between $400k and $700k | 22 | 0.6 | 1.0 | yes |
| condo with a gym | 3 | 0.4 | 0.4 | yes |
| 3+ bedrooms with a big yard for the kids | 23 | 0.8 | 0.8 | yes |
| energy-efficient home with solar panels | 0 | 0.0 | 0.0 | yes |
| 1 bedroom condo under $400k | 18 | 0.8 | 1.0 | yes |
| house with a deck and hiking access | 2 | 0.4 | 0.4 | yes |
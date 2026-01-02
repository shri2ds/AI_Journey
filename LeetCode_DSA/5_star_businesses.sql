#Top 5 States With 5 Star Businesses

WITH cte AS (
    SELECT state,
           COUNT(business_id) AS n_businesses,
           DENSE_RANK() OVER (ORDER BY COUNT(business_id) DESC) AS rnk
    FROM <table>
    WHERE stars = 5
    GROUP BY state
)
select
  state,
  n_businesses
from
  cte
where rnk <= 5

WITH daily_totals AS (
    SELECT 
        cust_id, 
        order_date, 
        SUM(total_order_cost) AS total
    FROM orders
    WHERE order_date BETWEEN '2019-02-01' AND '2019-05-01'
    GROUP BY cust_id, order_date
),
named_totals AS (
    SELECT 
        customers.first_name,
        daily_totals.total,
        daily_totals.order_date
    FROM daily_totals
    JOIN customers ON daily_totals.cust_id = customers.id
)
SELECT * 
FROM named_totals
ORDER BY total DESC
LIMIT 1;

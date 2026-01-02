SELECT DISTINCT business_name, 
    CASE 
        WHEN lower(business_name) LIKE '%restaurant%' 
             OR lower(business_name) LIKE '%restaurante%' 
             OR lower(business_name) LIKE '%restauranté%' 
        THEN 'restaurant'
        
        WHEN lower(business_name) LIKE '%café%' 
             OR lower(business_name) LIKE '%cafe%' 
             OR lower(business_name) LIKE '%coffee%' 
        THEN 'cafe'

        WHEN lower(business_name) LIKE '%school%' 
        THEN 'school'

        ELSE 'other'
    END AS business_type
FROM <table>

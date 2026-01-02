SELECT name
FROM Customer
WHERE coalesce(referee_id,'') != 2

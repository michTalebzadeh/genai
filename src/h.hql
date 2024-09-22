SELECT 
    countryincorporated1, 
    Proprietor1Address1, 
    COUNT(*) AS company_count 
FROM 
    ds.ocod_full_2024_03 
GROUP BY 
    countryincorporated1, 
    Proprietor1Address1 
HAVING 
    COUNT(*) > 1 
ORDER BY 
    company_count DESC;

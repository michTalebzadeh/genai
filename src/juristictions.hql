SELECT 
       t.countryincorporated1,
       t.NumberIncorporated,
       t.InvestmentTotalInMillions 
FROM (
      SELECT
          countryincorporated1,
          COUNT(countryincorporated1) AS NumberIncorporated,
          ROUND(SUM(pricePaid)/1000000,1) AS InvestmentTotalInMillions
      FROM 
          ds.ocod_full_2024_03
      GROUP BY
          countryincorporated1 
     ) t
ORDER BY
       t.InvestmentTotalInMillions
DESC
LIMIT 15
;


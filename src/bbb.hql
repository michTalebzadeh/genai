SELECT 
  district,
  FORMAT_NUMBER(NumberOfOffshoreOwned,0), -- Format as integer, no decimals
  FORMAT_NUMBER(total_price_in_billions_gbp,2), -- Format with 2 decimal places
  FORMAT_NUMBER(average_price_per_property,2) -- Format with 2 decimal places
FROM ds.bad_data
ORDER BY district;

WITH column_widths AS (
  SELECT
    MAX(LENGTH(district)) AS max_district_width,
    MAX(LENGTH(FORMAT_NUMBER(NumberOfOffshoreOwned,0))) AS max_owned_width,
    MAX(LENGTH(FORMAT_NUMBER(total_price_in_billions_gbp,2))) AS max_price_width,
    MAX(LENGTH(FORMAT_NUMBER(average_price_per_property,2))) AS max_avg_width
  FROM ds.bad_data
)
SELECT
  district,
  NumberOfOffshoreOwned,
  total_price_in_billions_gbp,
  average_price_per_property
FROM ds.bad_data
ORDER BY district
FETCH FIRST ROW ONLY WITH NO GAPS
CROSS JOIN column_widths
FORMAT '+-%-' ||  REPEAT('-', max_district_width + 2) || '-+-%-' || REPEAT('-', max_owned_width + 2) || '-+-%-' || REPEAT('-', max_price_width + 2) || '-+-%-' || REPEAT('-', max_avg_width + 2) || '-+' AS header_line
UNION ALL
SELECT
  district,
  FORMAT_STRING('%-' || max_district_width || 's', district),
  FORMAT_STRING('%-' || max_owned_width || 's', NumberOfOffshoreOwned),
  FORMAT_STRING('%-' || max_price_width || 's', total_price_in_billions_gbp),
  FORMAT_STRING('%-' || max_avg_width || 's', average_price_per_property)
FROM ds.bad_data
ORDER BY district;



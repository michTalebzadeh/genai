SELECT
    countryincorporated1,
    RPAD(standardized_address, 53, ' ') AS standardized_address,
    LPAD(CAST(CompaniesIncorpotated AS STRING), 5, ' ') AS CompaniesIncorpotated
FROM
(
SELECT
    countryincorporated1,
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                LOWER(Proprietor1Address1),
                'st[\. ]?helier', 'St Helier'),
            'esplandade|esplanande|esplande|esplanade', 'Esplanade'),
        '[\.,]', '')  AS standardized_address,
    COUNT(*) AS CompaniesIncorpotated
FROM
    ds.ocod_full_2024_03
WHERE
    countryincorporated1 = 'JERSEY'
GROUP BY
    countryincorporated1,
    REGEXP_REPLACE(
        REGEXP_REPLACE(
            REGEXP_REPLACE(
                LOWER(Proprietor1Address1),
                'st[\. ]?helier', 'St Helier'),
            'esplandade|esplanande|esplande|esplanade', 'Esplanade'),
        '[\.,]', '')
HAVING
    COUNT(*) > 100
) tmp
ORDER BY
    CompaniesIncorpotated DESC

;

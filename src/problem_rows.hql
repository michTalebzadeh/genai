DROP TABLE IF EXISTS ds.problem_rows;
CREATE TABLE ds.problem_rows
AS SELECT
     *
FROM
     ds.ocod_full_2024_03
WHERE 
     titlenumber
IN(
"276916",
"NGL30500A,1"
"350274",
"EX671482",
"NGL935756",
"NGL616483",
"WT103699",
"SY775740",
"NGL942065",
"ESX314508",
"NGL942065",
"NGL942066",
"GM330808",
"NGL915622",
"BGL55788",
"BK416904",
"NGL899979",
"CL156115"
)
;
DESC ds.problem_rows;
!exit

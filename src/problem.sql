/*
table name: Employee
+-------------+------+
| Column Name | Type |
+-------------+------+
| id          | int  |
| salary      | int  |
+-------------+------+
select the second highest salary from the Employee table.
*/

-- Solution:
select (
        select distinct
            salary
        from Employee
        order by salary desc
        limit 1
        offset
            1
    ) as SecondHighestSalary;

/*
same table, select the Nth highest salary from the Employee table.
*/

CREATE FUNCTION getNthHighestSalary(N INT) RETURNS 
INT 
BEGIN 
	SET N = N -1;
	RETURN (
	    SELECT DISTINCT
	        salary
	    FROM Employee
	    ORDER BY salary DESC
	    LIMIT 1
	    OFFSET
	        N
	);
END

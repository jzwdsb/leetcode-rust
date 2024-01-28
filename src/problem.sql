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
        order by salary
        limit 1
        offset
            1
    ) as SecondHighestSalary;

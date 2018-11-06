import json

import csv

employee_parsed = json.loads(employee_data)

emp_data = employee_parsed['employee_details']

# open a file for writing

employ_data = open('/tmp/EmployData.csv', 'w')

# create the csv writer object

csvwriter = csv.writer(employ_data)

count = 0

for emp in emp_data:

      if count == 0:

             header = emp.keys()

             csvwriter.writerow(header)

             count += 1

      csvwriter.writerow(emp.values())

employ_data.close()

import csv
import random

test_records = random.sample(range(1, 13612), 611)

with (
  open('dryBeanDataset.csv', 'r', newline='', encoding='utf-8-sig') as oldset,
  open('dryBeanTraining.csv', mode='w', newline='') as trainset,
  open('dryBeanTesting.csv', mode='w', newline='') as testset
  ):
  bean_reader = csv.reader(oldset, delimiter=',', dialect='excel')
  train_writer = csv.writer(trainset, delimiter=',', dialect='excel')
  test_writer = csv.writer(testset, delimiter=',', dialect='excel')
  line_count = 0
  for row in bean_reader:
    if line_count == 0:
      train_writer.writerow(row)
      test_writer.writerow(row)
    elif line_count in test_records:
      test_writer.writerow(row)
    else:
      train_writer.writerow(row)
    line_count += 1
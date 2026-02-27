from upload_to_dropbox import *


year = "2026"   # Change this to the current year

months = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

dropbox_path = f"Video_Recordings/Bee_Recordings_{year}"

test_data = "test.txt"

# Make month directories, then upload test data to it
for month in months:
    month_dropbox_path = dropbox_path + "/" + month
    upload_to_dropbox(test_data, month_dropbox_path)


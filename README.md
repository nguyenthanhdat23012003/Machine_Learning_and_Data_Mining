# SVMSentiment

Install dependencies:

- (Optional) Create virtual environment `venv`

```
$ python3 -m venv venv
$ source venv/Scipts/activate (Window)
$ source venv/bin/activate
```

- Install requirement libraries

```
$ pip3 install -r requirements.txt
```

Run all code in file main.ipynb by jupyte notebook to export model and evaluate

To save time training the model, you can use a smaller dataset by reducing the number of lines in the dataset, in this code we are only using 30000 lines, you can change the number of lines in the main file, find the line with this content "df = df.sample(n=30000)" and change the number 30000 to the number of rows you want, knowing that the dataset has 810820 rows

Change value of test in file model_test.ipynb and run all code to test model

Thanks

# CVClassroom
Classroom-based Structure for CV Training (based on Noisy Student Training)

- Multi-Teacher?
- Multi-Student?
- BOTH?

stay tuned!

## Reproducibility notes 

Most files in this project are in keras/tensorflow. 

### Part 1: Stanford Cars Dataset 
1. download the zip file of the dataset here: [https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset?resource=download] 
This is what the folder should look like after you extract: 
![Image of folder of extracted dataset from kaggle](<images/image_of_extracted_dataset_from_kaggle.jpg>)

2. Move the files into the stanford cars folder, as seen below: 
!["folder named stanford_cars_dataset"](<images/dataset_folder_name.jpg>)


In the end, this is what the folder should look like: 
![Image of folder after dataset extraction. It contains two sub-folders called "cars_test" and "cars_train", two .csv files called "test.csv" and "train.csv", and 2 .mat files called "cars_annos.mat" and "cars_test_annos_withlabels (1).mat"](<images/final_folder_image.png>) 
Note that cars_annos.mat is unnecessary and can be removed. 


Sources of files in the folder: 

- The CSV files of some labels of the dataset were from this link: [https://github.com/BotechEngineering/StanfordCarsDatasetCSV] 
- More labels of cars were downloaded (the file called "cars_test_annos_withlabels (1).mat") from this kaggle link: [https://www.kaggle.com/datasets/abdelrahmant11/standford-cars-dataset-meta] 




### Part 2: The Car Connection Picture Dataset (somewhat unlabelled)
1. Download the zip file of the dataset here: [https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset?resource=download] 
The folder should consist of only .png files 
Name the folder "imgs", and put the "imgs" folder under the car_connection_dataset folder in this repository. 


### Part 3: teacher models 
For the transfer learning, we need a different structure of the stanford cars dataset. Run download_stanford_cars_dataset.py to download this other structure. 



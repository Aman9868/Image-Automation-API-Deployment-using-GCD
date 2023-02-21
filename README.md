
# Image Automation

The project focuses on automates the images with the helpo of Computer Vision & Deep Learning.

The objective of the projects is to automates task like:

1. Image Classification
2. Image Segmentation
3. Object detection
4. Image Captioning
5. Image Super Resolution
6. Key Point Detection
7. Low Light Image Enhancement.
8. Panoptic Segmentation.
9. Image Visual Question Answering.
10. Image to Ocr.


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`pip install -r requirements.txt`



## Run Locally

Clone the project

```bash
  git rep clone Aman9868/Image-Automation-API-Deploymnet-using-GCD
```

Go to the project directory

```bash
  cd Image-Automation-API-Deploymnet
```

Run app

```bash
  flask --app routes run
  Running on http://127.0.0.1:5000
```




## Deployment on GCD

To deploy this follow these steps:

```bash
  1. Open GCD Cloud Shell.
  2. Type in terminal : https://github.com/Aman9868/Image-Automation-API-Deploymnet-using-GCD.git
  3. Create Global Environment Variable in Terminal : export PROJECT_ID = kubernets-test-281207  //<project_name>
  4. Build Image : docker build -t gcr.io/${PROJECT_ID}/image-app:v1 .
  5. Check image: docker images
  6. Authenticate image on Container registry : gcloud auth configure-docker.io
  7. Push Image : docker push gcr.io/${PROJECT_ID}/image-app:v1
  8. Create Engine zone : gcloud config set compute/zone us-central1
  9. Create Cluster : gcloud container clusters create imageapp-cluster --num-nodes=2
  10. Deploy Application : kubectl create deployment image-app --image=gcr.io/${PROJECT_ID}/image-app:v1
  11. Expose Application : kubectl expose deployment image-app --type=LoadBalancer --port 80 --target-port 8080
  12. Check Service : kubectl get service
  13. Run Service : Copy External IP & paste it into your browser then app run succesfully.


```


## Screenshots
![Screenshot 2023-02-21 095906](https://user-images.githubusercontent.com/60923869/220248248-5f9a16e9-2eef-4256-bcb8-6f79b961232a.png)

![Screenshot 2023-02-21 100001](https://user-images.githubusercontent.com/60923869/220248265-37c52a9c-2910-40ed-b046-55ec70c6a8e8.png)





## Demo
1. Object Detection

![Screenshot_2023-01-16_163640](https://user-images.githubusercontent.com/60923869/220245625-51f097a2-793f-4521-97a3-991144de81a5.jpg)

2. Panoptic Segmentation

![14](https://user-images.githubusercontent.com/60923869/220246409-20731aba-0ccb-4b7d-a119-dff8ae10dfeb.jpg)



## Authors

- [@Aman9868](https://www.github.com/Aman9868)








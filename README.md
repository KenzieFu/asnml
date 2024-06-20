## Installation

To run this app on your device, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/KenzieFu/asnml.git

2. Open the project in Visual Studio Code

3. Install dependencies
   ```bash
   pip install -r requirentments.txt
4. Create service account for fetching the model in database
5. move that service account (.json file) to this directory and on your .env specified your GOOGLE APPLICATION CREDENTIALS with your service account path that you downloaded

   
![e](https://github.com/KenzieFu/asnml/assets/95515953/8aa44574-9d47-49fe-a80a-30cedf2e8fb9)

6. run the application
   ```bash
   uvicorn main:app --reload

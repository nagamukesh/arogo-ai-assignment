// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('mentalHealthForm');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Get form data
        const formData = new FormData(form);
        const data = {};
        
        // Convert form data to JSON object
        formData.forEach((value, key) => {
            data[key] = ['school_year', 'age', 'height', 'weight', 'phq_score', 'gad_score', 'epworth_score']
                .includes(key) ? Number(value) : value;
        });

        // Make the POST request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(result => {
            console.log('Received result:', result);
            
            // Display the prediction and explanation
            document.getElementById('prediction').textContent = 'Prediction: ' + result.prediction;
            document.getElementById('explanation').textContent = 'Explanation: ' + result.explanation;
            
            // Show the result container
            document.getElementById('result').style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred: ' + error.message);
        });
    });
});








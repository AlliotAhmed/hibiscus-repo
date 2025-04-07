document.addEventListener('DOMContentLoaded', function() {
    // Get the upload form and loading indicator
    const uploadForm = document.getElementById('upload-form');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    // Show loading indicator when form is submitted
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            // Hide the form
            uploadForm.classList.add('d-none');
            
            // Show loading indicator
            loadingIndicator.classList.remove('d-none');
        });
    }
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Function to preview the selected image before upload
    const fileInput = document.getElementById('formFile');
    if (fileInput) {
        fileInput.addEventListener('change', function(event) {
            // You could add image preview functionality here if desired
            // For now, just confirm a file was selected
            const file = event.target.files[0];
            if (file) {
                console.log('File selected:', file.name);
            }
        });
    }
    
    // Add event listener to close alert messages
    var alertList = document.querySelectorAll('.alert');
    alertList.forEach(function (alert) {
        new bootstrap.Alert(alert);
    });
});

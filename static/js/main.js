// Main JavaScript file for Hibiscus Leaf Disease Classifier

// Enable Bootstrap validation
(function () {
    'use strict'
    
    // Fetch all forms that need validation
    var forms = document.querySelectorAll('.needs-validation')
    
    // Loop over them and prevent submission
    Array.prototype.slice.call(forms)
        .forEach(function (form) {
            form.addEventListener('submit', function (event) {
                if (!form.checkValidity()) {
                    event.preventDefault()
                    event.stopPropagation()
                }
                
                form.classList.add('was-validated')
            }, false)
        })
})()

// Function to handle drag and drop file upload
function setupFileUpload() {
    const uploadArea = document.querySelector('.upload-area');
    
    if (!uploadArea) return;
    
    const fileInput = document.querySelector('#formFile');
    
    // Prevent default behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }
    
    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        
        // Trigger file input change event
        fileInput.dispatchEvent(new Event('change'));
    }
    
    // Update file name display on file selection
    fileInput.addEventListener('change', function() {
        const fileNameDisplay = document.querySelector('#fileName');
        if (fileNameDisplay) {
            if (this.files.length > 0) {
                fileNameDisplay.textContent = this.files[0].name;
                // Show the upload button
                document.querySelector('#uploadBtn').classList.remove('d-none');
            } else {
                fileNameDisplay.textContent = 'No file selected';
                document.querySelector('#uploadBtn').classList.add('d-none');
            }
        }
    });
}

// Show file preview when selected
function setupFilePreview() {
    const fileInput = document.querySelector('#formFile');
    
    if (!fileInput) return;
    
    fileInput.addEventListener('change', function() {
        const previewContainer = document.querySelector('#imagePreview');
        if (!previewContainer) return;
        
        if (this.files && this.files[0]) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                previewContainer.innerHTML = `
                    <div class="text-center mb-3">
                        <img src="${e.target.result}" class="img-fluid" style="max-height: 300px;" alt="Image preview">
                    </div>
                `;
            }
            
            reader.readAsDataURL(this.files[0]);
        }
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    setupFileUpload();
    setupFilePreview();
});

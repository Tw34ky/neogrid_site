// Typing animation for search placeholder
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-term');
    if (!searchInput) return;
    
    // Messages to cycle through (customize as needed)
    const messages = [
        "Задайте вопрос по вашим данным...",
        // "Например: какие файлы изменены сегодня?",
        "Попробуйте: найти документы за март...",
        "Спросите: где мои последние загрузки?"
    ];
    
    let messageIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    let typingSpeed = 50;
    const originalPlaceholder = searchInput.placeholder;
    
    function typeEffect() {
        const currentMessage = messages[messageIndex];
        
        if (isDeleting) {
            // Deleting text
            searchInput.placeholder = currentMessage.substring(0, charIndex - 1);
            charIndex--;
            typingSpeed = 25; // Faster when deleting
        } else {
            // Typing text
            searchInput.placeholder = currentMessage.substring(0, charIndex + 1);
            charIndex++;
            typingSpeed = 50; // Normal typing speed
        }
        
        // Check if finished typing or deleting
        if (!isDeleting && charIndex === currentMessage.length) {
            // Pause before deleting
            isDeleting = true;
            typingSpeed = 1500; // Longer pause at end of message
        } else if (isDeleting && charIndex === 0) {
            // Move to next message
            isDeleting = false;
            messageIndex = (messageIndex + 1) % messages.length;
            typingSpeed = 500; // Short pause before typing next
        }
        
        setTimeout(typeEffect, typingSpeed);
    }
    
    // Start the typing effect after a short delay
    setTimeout(typeEffect, 1000);
    
    // Optional: Pause animation when input is focused
    searchInput.addEventListener('focus', function() {
        // You can add logic here to pause if needed
    });
});

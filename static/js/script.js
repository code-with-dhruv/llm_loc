document.getElementById('sendButton').addEventListener('click', async () => {
    const query = document.getElementById('questionInput').value;
    if (!query) {
        alert('Please enter a query');
        return;
    }
    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        const data = await response.json();
        console.log(data); // Handle and render search results here
    } catch (error) {
        console.error('Error:', error);
    }
});

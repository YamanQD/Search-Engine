function search() {
    const query = document.getElementById('search-input').value;
    if (!query) {
        alert('Please enter a query.');
        return;
    }

    fetch(`http://127.0.0.1:8000/search?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => displayResults(data))
        .catch(error => console.error('Error fetching data:', error));
}

function displayResults(results) {
    const resultsContainer = document.getElementById('results');
    resultsContainer.innerHTML = '';

    for (const key in results) {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';

        const title = document.createElement('h3');
        title.textContent = results[key];
        resultItem.appendChild(title);

        const body = document.createElement('p');
        body.textContent = `Document ID: ${key}`;
        resultItem.appendChild(body);

        resultsContainer.appendChild(resultItem);
    }
}

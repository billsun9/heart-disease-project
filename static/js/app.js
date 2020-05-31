const varNames = document.querySelectorAll('.var-name');
console.log(varNames);

varNames.forEach(varName => {
    varName.addEventListener('click', () => {
        let info = varName.nextElementSibling;
        info.classList.toggle('show');
    });
});
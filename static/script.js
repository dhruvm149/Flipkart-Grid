const bar = document.getElementById('bar');
const close = document.getElementById('close');
const nav = document.getElementById('navbar');

if (bar) {
    bar.addEventListener('click',() =>{
        nav.classList.add('active');
    })
}
if (close) {
    close.addEventListener('click',() =>{
        nav.classList.remove('active');
    })
}

/* Bottom to Top button */

const toTop = document.querySelector(".to-top");

window.addEventListener("scroll", () => {
    if(window.pageYOffset > 100) {
        toTop.classList.add("active");
    }else{
        toTop.classList.remove("active");
    }
})

var removeCartItembtn = document.getElementsByClassName('remove')
console.log(removeCartItembtn)
for(var i=0; i<removeCartItembtn.length; i++){
    var button = removeCartItembtn[i]
    button.addEventListener('click', function(event){
        var buttonClicked = event.target
        buttonClicked.parentElement.parentElement.parentElement.parentElement.remove()
        updateCartTotal()
    })
}

var quantityInputs = document.getElementsByClassName('quantity')
for(var i=0; i<quantityInputs.length; i++){
    var input = quantityInputs[i]
    input.addEventListener('change', quantitychanged)
}

var cartbtn = document.getElementsByClassName('btn-cart')
for(var i=0; i<cartbtn.length; i++){
    var input = cartbtn[i]
    input.addEventListener('click', addToCartClicked)  
}


function quantitychanged(event){
    var input = event.target
    if(isNaN(input.value) || input.value <= 0){
     input.value = 1
    }
    updateCartTotal()
}

function updateCartTotal(){
    var cartRows = document.getElementsByClassName('cart-row')
    var total = 0;
    var total1 = 0;
    for(i=0; i<cartRows.length; i++){
        var cartRow = cartRows[i]
        var priceElement = cartRow.getElementsByClassName('price')[0]
        var quantityElement = cartRow.getElementsByClassName('quantity')[0]
        var price = parseFloat(priceElement.innerText.replace('$',''))
        var quantity = +quantityElement.value

        total = total+(price * quantity)
        }
        total1 = total
        total = total+35
    document.getElementsByClassName('cart-total-price')[0].innerText = '$' + total
    document.getElementsByClassName('cart-subtotal-price')[0].innerText = '$'+total1
}
function addToCartClicked(event){
    
    var btnn= event.target
    var pdt =btnn.parentElement
    var pdtname = pdt.getElementsByClassName('name')[0].innerText
    var pdtprice = pdt.getElementsByClassName('cost')[0].innerText
    var img = pdt.getElementsByClassName('img')[0].src
    console.log(pdtname, pdtprice, img)
    addItemToCart(pdtname, pdtprice, img)
    updateCartTotal()
}
function addItemToCart(pdtname, pdtprice, img){
    var cartRow= document.createElement('tr')

    var cartItems = document.getElementsByClassName('table')[0]

    var cartRowContents =`
        
          <td>
            <div class="cart-info">
              <img src="${img}" width="120" height="120" alt="Tshirt" />
              <div>
                <p>${pdtname}</p>
                <small>Price: ${pdtprice}</small>
                <br />
                <button class="remove">Remove</a></button>
              </div>
            </div>
          </td>
          <td><input class="quantity" type="number" value="1"  width="50px" height="30px" /></td>
          <td class="price">${pdtprice}</td>
    `
    cartRow.innerHTML = cartRowContents
    cartRow.classList.add('cart-row')
    cartItems.append(cartRow)
    cartRow.getElementsByClassName('remove')[0].addEventListener('click', function(event){
        var buttonClicked = event.target
        buttonClicked.parentElement.parentElement.parentElement.parentElement.remove()
        updateCartTotal()
    })
        cartRow.getElementsByClassName('quantity')[0].addEventListener('change',quantitychanged)

    }

    const searchInput = document.getElementById('search');
    const autocompleteResults = document.getElementById('autocompleteResults');

    searchInput?.addEventListener('input', async () => {
        const inputValue = searchInput.value.trim();

        if (inputValue.length === 0) {
            autocompleteResults.style.display = 'none';
            autocompleteResults.innerHTML = '';
            return;
        }

        const apiUrl = "http://127.0.0.1:5000/search";
        const body = {
            "productName": document.getElementById("search").value
        };
        
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(body)
            });
            const data = await response.json();

            if (data.length > 0) {
                autocompleteResults.style.display = 'block';
                autocompleteResults.innerHTML = '';

                data.forEach(item => {
                    const resultItem = document.createElement('div');
                    resultItem.textContent = item.product_title;
                    resultItem.classList.add('resultItem');
                    
                    resultItem.addEventListener('click', () => {
                        window.location.href = "/shop";

                        searchInput.value = item.product_title;
                        autocompleteResults.style.display = 'none';
                    });
                    
                    autocompleteResults.appendChild(resultItem);
                });
            } else {
                autocompleteResults.style.display = 'none';
                autocompleteResults.innerHTML = '';
            }
        } catch (error) {
            console.error('API Error:', error);
        }
    });

    document.addEventListener('click', event => {
        if (!autocompleteResults.contains(event.target) && event.target !== searchInput) {
            autocompleteResults.style.display = 'none';
        }
    });
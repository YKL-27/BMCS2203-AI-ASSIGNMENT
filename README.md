# AI Assignment - Chatbot
## 2025-12-14 Submit
<h3>Click <a href="QUESTION.md">HERE</a> for question requirement</h3>

### Software Requirements
- Python 3.12.8

### How to open this project
1. Download this project
2. Go to terminal, enter <code>pip install -r requirements.txt</code>
3. Run training.py to train data stored in 'data' folder, and save joblib files in 'joblib' folder.
4. Run testing.py to test model locally
<br>*The streamlit app runs main.py, and uses joblib files stored in GitHub

### Report
https://docs.google.com/document/d/1WS0LGiR7hrceA38k9jtDs2pStxVTjxA3/edit?usp=sharing&ouid=118013344239763753061&rtpof=true&sd=true

### Background
- This is a chatbot of for Astra Imperium Hotel (get it? the initials are 'AI' lol), a hotel located in the heart of Kuala Lumpur. 
- The chatbot is built in the Hotel main website, which allow customers to ask question easily.

### Intents
#### 25 intents from dataset retrieved online
- invoices
- cancellation_fees
- check_in
- check_out
- customer_service
- human_agent
- host_event
- file_complaint
- leave_review
- book_hotel
- cancel_hotel_reservation
- change_hotel_reservation
- check_hotel_facilities
- check_hotel_offers
- check_hotel_prices
- check_hotel_reservation
- search_hotel
- store_luggage
- check_menu
- add_night
- book_parking_space
- bring_pets
- redeem_points
- get_refund
- shuttle_service
#### 10 new custom intents
- greeting
- check_functions
- check_room_type
- check_room_availability
- check_nearby_attractions
- check_child_policy
- check_smoking_policy
- check_payment_methods
- check_lost_item
- goodbye
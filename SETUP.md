# KrishiNetra – MySQL Setup Guide

## Files delivered
| File | Purpose |
|------|---------|
| `krishinetra_schema.sql` | Pure SQL — create all tables + seed data |
| `app_mysql.py` | Flask app with SQLAlchemy wired to MySQL |
| `.env.example` | Environment variable template |

---

## Step 1 – Install MySQL (if not installed)

### macOS
```bash
brew install mysql
brew services start mysql
mysql_secure_installation
```

### Ubuntu / Debian
```bash
sudo apt update && sudo apt install mysql-server
sudo systemctl start mysql
sudo mysql_secure_installation
```

### Windows
Download MySQL Installer from https://dev.mysql.com/downloads/installer/

---

## Step 2 – Create the database

```bash
mysql -u root -p
```
Inside the MySQL shell:
```sql
CREATE DATABASE krishinetra_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'krishiuser'@'localhost' IDENTIFIED BY 'StrongPass123!';
GRANT ALL PRIVILEGES ON krishinetra_db.* TO 'krishiuser'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

---

## Step 3 – Run the schema

```bash
mysql -u krishiuser -p krishinetra_db < krishinetra_schema.sql
```

Verify tables were created:
```bash
mysql -u krishiuser -p krishinetra_db -e "SHOW TABLES;"
```

Expected output:
```
crops
crop_scans
crop_suggestion_requests
crop_suggestion_results
farms
fertilizer_products
order_items
orders
saved_crop_plans
users
weather_logs
yield_predictions
```

---

## Step 4 – Configure environment

```bash
cp .env.example .env
# Edit .env and set your DB_USER, DB_PASS etc.
```

---

## Step 5 – Install Python dependencies

```bash
pip install flask flask-cors flask-sqlalchemy pymysql \
            scikit-learn numpy pillow python-dotenv
```

---

## Step 6 – Run the app

```bash
# Load .env automatically
python -c "from dotenv import load_dotenv; load_dotenv()" 2>/dev/null || true
python app_mysql.py
```

Or set env vars inline:
```bash
DB_USER=krishiuser DB_PASS=StrongPass123! python app_mysql.py
```

Server starts at: **http://localhost:5000**

---

## Step 7 – Test the endpoints

### Yield Prediction (saves to `yield_predictions` table)
```bash
curl -X POST http://localhost:5000/api/predict-yield \
  -H "Content-Type: application/json" \
  -H "X-User-Id: 1" \
  -d '{"crop":"rice","temperature":28,"rainfall":850,
       "humidity":65,"soil_ph":6.5,"nitrogen":120,
       "phosphorus":60,"area":2.5}'
```

### Crop Scan (saves to `crop_scans` table)
```bash
curl -X POST http://localhost:5000/api/scan-crop \
  -H "X-User-Id: 1" \
  -F "image=@/path/to/crop.jpg"
```

### Payment (saves to `orders` + `order_items` tables)
```bash
curl -X POST http://localhost:5000/api/payment \
  -H "Content-Type: application/json" \
  -H "X-User-Id: 1" \
  -d '{"total":1700,"method":"UPI","upi_id":"farmer@upi",
       "items":[{"id":"npk-1","name":"NPK 20-20-20","price":850,"qty":2}],
       "delivery_name":"Ravi Kumar","delivery_phone":"9876543210",
       "delivery_address":"Yelahanka, Bangalore"}'
```

---

## Database schema overview

```
users ──< farms
users ──< yield_predictions >── farms
users ──< crop_scans >── farms
users ──< crop_suggestion_requests >── crop_suggestion_results
users ──< orders >── order_items >── fertilizer_products
users ──< weather_logs
crops (reference table — seeded)
fertilizer_products (reference table — seeded)
```

### Note on authentication
`X-User-Id` header is a placeholder. For production, replace `get_user_id()` in
`app_mysql.py` with a proper JWT/session check (e.g. Flask-JWT-Extended).

---

## Docker Compose (optional, full stack)

```yaml
version: "3.9"
services:
  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpass
      MYSQL_DATABASE: krishinetra_db
      MYSQL_USER: krishiuser
      MYSQL_PASSWORD: StrongPass123!
    ports: ["3306:3306"]
    volumes:
      - ./krishinetra_schema.sql:/docker-entrypoint-initdb.d/schema.sql

  api:
    build: .
    command: python app_mysql.py
    environment:
      DB_USER: krishiuser
      DB_PASS: StrongPass123!
      DB_HOST: db
      DB_PORT: 3306
      DB_NAME: krishinetra_db
    ports: ["5000:5000"]
    depends_on: [db]
```

Start everything:
```bash
docker compose up --build
```

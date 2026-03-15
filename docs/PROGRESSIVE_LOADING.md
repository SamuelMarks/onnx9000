# Serving Progressive ONNX Models

To utilize the HTTP 206 Partial Content features of `onnx9000` progressive loader, your web server must support HTTP Range requests and CORS.

## Nginx Configuration

```nginx
server {
    listen 80;
    server_name models.example.com;

    location / {
        root /var/www/models;
        
        # Step 075: Enable CORS
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Range, Content-Type';
        add_header 'Access-Control-Expose-Headers' 'Content-Length, Content-Range, Accept-Ranges';

        # Ensure byte-range requests are enabled
        max_ranges 10;
        
        # Step 076: Enable Gzip for metadata (optional, but good)
        gzip on;
        gzip_types application/octet-stream application/x-protobuf;
    }
}
```

## Apache Configuration

```apache
<VirtualHost *:80>
    DocumentRoot "/var/www/models"
    ServerName models.example.com

    <Directory "/var/www/models">
        Header set Access-Control-Allow-Origin "*"
        Header set Access-Control-Allow-Methods "GET, OPTIONS"
        Header set Access-Control-Allow-Headers "Range, Content-Type"
        Header set Access-Control-Expose-Headers "Content-Length, Content-Range, Accept-Ranges"
    </Directory>
</VirtualHost>
```

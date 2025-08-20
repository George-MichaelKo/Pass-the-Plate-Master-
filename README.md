# Pass the Plate

A Next.js application that empowers communities to share surplus food, reduce waste, and combat food insecurity.

## Features

- **AI-powered smart pantry** – Scan receipts to auto-fill items with expiration tracking.
- **Trust & safety tools** – ID verification, “Verified Giver” badges, and pickup hubs.
- **Modern UI** – Built with Tailwind CSS, responsive design, and dark mode support.
- **Seamless experience** – Form validation with Zod, React Hook Form, and toast notifications.
- **Interactive maps** – Google Maps integration for drop-off and pickup locations.

> 💡 Project design and features were **reviewed by engineers and designers at Amazon during the 2025 Codepath x Amazon Next Design Challenge**.

## Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/pass-the-plate.git
   cd pass-the-plate
```

2. Install dependencies
   ```bash
   pnpm install
```

3. Create a .env.local file in the root directory and add your Google Maps API key:

   ## Google Maps Setup (Beginner-Friendly)
          
   To use the Google Maps features in Pass the Plate, follow these steps:
          
          a. **Create a Google Cloud account**  
             Go to [Google Cloud Console](https://console.cloud.google.com/) and sign in or create a new account.
          
          b. **Create a new project**  
             - Click the project dropdown in the top menu.  
             - Select "New Project" and give it a name.  
          
          c. **Enable APIs**  
             Go to **APIs & Services > Library** and enable the following APIs for your project:  
             - Maps JavaScript API  
             - Places API (for location autocomplete and place details)  
             - Geocoding API (to convert addresses to coordinates)  
          
          d. **Generate an API key**  
             - Go to **APIs & Services > Credentials**.  
             - Click "Create Credentials" > "API key".  
             - Copy the key.
          
          e. **Add the key to your project**  
             Create a `.env.local` file in the root of the repository and add:  
             ```bash
              NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
           ```

4. Start the development server:
```bash
pnpm dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Tech Stack

- [Next.js](https://nextjs.org/)
- [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Zod](https://zod.dev/)
- [React Hook Form](https://react-hook-form.com/)
- [Google Maps API](https://developers.google.com/maps)
- [Radix UI](https://www.radix-ui.com/)
- [Lucide Icons](https://lucide.dev/)


## Project Structure

```
pass-the-plate/
├── app/                 # Next.js app directory
├── components/          # React components
├── context/             # React context providers
├── lib/                 # Utility functions
├── public/              # Static assets
└── styles/              # Global styles
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

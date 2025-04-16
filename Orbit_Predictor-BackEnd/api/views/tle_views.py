import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
import logging

logger = logging.getLogger(__name__)

class TleProxyView(APIView):
    """
    Proxy view for fetching TLE data from external sources.
    This avoids CORS issues by having the backend server make the request.
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request, norad_id=None):
        if not norad_id:
            return Response(
                {"error": "NORAD ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # NASA API key
        NASA_API_KEY = "bFDWy9zOJ2zOnjLbd8rz0eZtVcWwkyT4XUrC3II5"
        
        # Try multiple sources for TLE data
        sources = [
            # NASA's TLE API 
            {
                "url": f"https://tle.ivanstanojevic.me/api/tle/{norad_id}",
                "headers": {
                    "Accept": "application/json",
                    "User-Agent": "SatelliteCollisionPredictor/1.0",
                    "X-Api-Key": NASA_API_KEY
                },
                "parser": self._parse_nasa_response
            },
            # CelesTrak as fallback
            {
                "url": f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE",
                "headers": {},
                "parser": self._parse_celestrak_response
            }
        ]
        
        for source in sources:
            try:
                print(f"DEBUG: Attempting to fetch TLE data from {source['url']}")
                print(f"DEBUG: Headers: {source['headers']}")
                response = requests.get(
                    source["url"],
                    headers=source["headers"],
                    timeout=5  # 5 second timeout
                )
                
                print(f"DEBUG: Response status: {response.status_code}")
                print(f"DEBUG: Response headers: {response.headers}")
                
                if response.status_code == 200:
                    print(f"DEBUG: Response content (first 200 chars): {response.text[:200]}")
                    parsed_data = source["parser"](response)
                    if parsed_data:
                        print(f"DEBUG: Successfully parsed data from {source['url']}")
                        print(f"DEBUG: Parsed data: {parsed_data}")
                        return Response(parsed_data)
            
            except Exception as e:
                print(f"DEBUG: Error fetching TLE data from {source['url']}: {str(e)}")
                # Specific handling for the NASA API which sometimes has connection issues
                if "tle.ivanstanojevic.me" in source["url"]:
                    try:
                        # Try a direct fetch with minimal error handling
                        import urllib.request
                        req = urllib.request.Request(
                            source["url"],
                            headers=source["headers"]
                        )
                        with urllib.request.urlopen(req, timeout=5) as response:
                            data = response.read().decode("utf-8")
                            # Check if it looks like valid JSON
                            if '{"@context":' in data and '"line1":' in data and '"line2":' in data:
                                print(f"DEBUG: Successfully retrieved TLE data using urllib")
                                import json
                                tle_data = json.loads(data)
                                parsed_data = {
                                    "name": tle_data.get("name", f"Satellite {tle_data.get('satelliteId')}"),
                                    "line1": tle_data["line1"],
                                    "line2": tle_data["line2"],
                                    "source": "nasa_api_direct"
                                }
                                print(f"DEBUG: Parsed NASA data: {parsed_data}")
                                return Response(parsed_data)
                    except Exception as inner_e:
                        print(f"DEBUG: Secondary fetch attempt also failed: {str(inner_e)}")
                
                # Continue to next source if both attempts fail
                continue
        
        # If all sources fail, return demo data
        demo_data = self._get_demo_tle(norad_id)
        return Response(demo_data)
    
    def _parse_nasa_response(self, response):
        """Parse response from NASA TLE API"""
        try:
            data = response.json()
            if "line1" in data and "line2" in data:
                return {
                    "name": data.get("name", f"Satellite {data.get('satelliteId')}"),
                    "line1": data["line1"],
                    "line2": data["line2"],
                    "source": "nasa_api"
                }
        except Exception as e:
            logger.error(f"Error parsing NASA API response: {str(e)}")
        return None
    
    def _parse_celestrak_response(self, response):
        """Parse response from CelesTrak"""
        try:
            text = response.text
            lines = text.strip().split('\n')
            if len(lines) >= 2:
                name = lines[0].strip() if len(lines) >= 3 else f"Satellite {lines[1].split()[1]}"
                return {
                    "name": name,
                    "line1": lines[-2].strip(),
                    "line2": lines[-1].strip(),
                    "source": "celestrak"
                }
        except Exception as e:
            logger.error(f"Error parsing CelesTrak response: {str(e)}")
        return None
    
    def _get_demo_tle(self, norad_id):
        """Provide demo TLE data based on satellite type"""
        satellite_id = str(norad_id)
        
        # Use appropriate demo orbits based on satellite type
        if satellite_id == "25544":  # ISS
            return {
                "name": "ISS (ZARYA) [DEMO]",
                "line1": "1 25544U 98067A   23136.55998435  .00010491  00000+0  19297-3 0  9994",
                "line2": "2 25544  51.6412 238.8846 0006203  86.3288 273.8410 15.50132295345582",
                "source": "demo"
            }
        elif satellite_id.startswith("4"):  # Starlink
            return {
                "name": f"STARLINK [DEMO] {satellite_id}",
                "line1": "1 44932U 19074A   23135.53046881  .00009706  00000+0  61808-4 0  9994",
                "line2": "2 44932  53.0504  59.9061 0001038  89.1133 270.9984 15.06390133156550",
                "source": "demo"
            }
        elif satellite_id == "20580":  # Hubble
            return {
                "name": "HUBBLE [DEMO]",
                "line1": "1 20580U 90037B   23136.29579028  .00000612  00000+0  32151-4 0  9992",
                "line2": "2 20580  28.4698 287.8908 0002449 156.5490 203.5557 15.09819891423633",
                "source": "demo"
            }
        elif satellite_id.startswith("3"):  # NOAA
            return {
                "name": f"NOAA [DEMO] {satellite_id}",
                "line1": "1 33591U 09005A   23135.50377627  .00000039  00000+0  51303-4 0  9992",
                "line2": "2 33591  99.1286 139.9440 0014096  92.7583 267.5211 14.12514767732624",
                "source": "demo"
            }
        else:  # Generic LEO
            return {
                "name": f"DEMO_ORBIT_{satellite_id}",
                "line1": "1 99999U 98067A   23136.55998435  .00010491  00000+0  19297-3 0  9994",
                "line2": "2 99999  51.6412 238.8846 0006203  86.3288 273.8410 15.50132295345582",
                "source": "demo"
            }
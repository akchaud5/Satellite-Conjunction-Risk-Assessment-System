"""
URL configuration for orbit_predictor project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.http import JsonResponse, HttpResponseRedirect

def api_root(request):
    """Root URL view that provides information about the API"""
    return JsonResponse({
        "message": "Welcome to the Satellite Conjunction Risk Assessment System API",
        "version": "1.0",
        "endpoints": {
            "API Root": "/api/",
            "Login": "/api/login/",
            "Register": "/api/register/",
            "CDMs": "/api/cdms/",
            "Collision Data": "/api/collisions/"
        }
    })

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('', api_root, name='api_root'),  # Root URL handler
    path('api/', include('api.urls')), 
]

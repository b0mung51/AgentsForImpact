import threading


def get_directions(destination: str, origin: str | None = None) -> str:
    try:
        return _get_directions_mapkit(destination, origin)
    except Exception:
        return _get_directions_fallback(destination, origin)


def _get_directions_mapkit(destination: str, origin: str | None = None) -> str:
    import CoreLocation
    import MapKit

    result: dict[str, object] = {"error": None, "steps": None}
    done = threading.Event()

    geocoder = CoreLocation.CLGeocoder.alloc().init()

    def on_dest_geocoded(placemarks, error):
        if error or not placemarks:
            result["error"] = f"Could not find destination: {destination}"
            done.set()
            return

        dest_pm = placemarks[0]
        dest_item = MapKit.MKMapItem.alloc().initWithPlacemark_(
            MapKit.MKPlacemark.alloc().initWithPlacemark_(dest_pm)
        )

        if origin:
            def on_origin_geocoded(origin_placemarks, origin_error):
                if origin_error or not origin_placemarks:
                    result["error"] = f"Could not find origin: {origin}"
                    done.set()
                    return
                origin_pm = origin_placemarks[0]
                origin_item = MapKit.MKMapItem.alloc().initWithPlacemark_(
                    MapKit.MKPlacemark.alloc().initWithPlacemark_(origin_pm)
                )
                _calculate(origin_item, dest_item)

            geocoder.geocodeAddressString_completionHandler_(origin, on_origin_geocoded)
        else:
            origin_item = MapKit.MKMapItem.mapItemForCurrentLocation()
            _calculate(origin_item, dest_item)

    def _calculate(source_item, destination_item):
        request = MapKit.MKDirections.Request.alloc().init()
        request.setSource_(source_item)
        request.setDestination_(destination_item)
        request.setTransportType_(1)  # walking

        directions = MapKit.MKDirections.alloc().initWithRequest_(request)

        def on_calc(response, error):
            if error:
                result["error"] = f"Could not calculate directions: {error}"
                done.set()
                return

            routes = response.routes()
            if not routes:
                result["error"] = "No route found."
                done.set()
                return

            route = routes[0]
            steps: list[str] = []
            for step in route.steps():
                instruction = step.instructions()
                distance_m = float(step.distance())
                if instruction:
                    if distance_m > 0:
                        feet = int(distance_m * 3.281)
                        steps.append(f"{instruction} for {feet} feet")
                    else:
                        steps.append(str(instruction))

            total_feet = int(float(route.distance()) * 3.281)
            total_minutes = int(float(route.expectedTravelTime()) / 60)
            result["steps"] = {
                "steps": steps,
                "total_feet": total_feet,
                "total_minutes": total_minutes,
            }
            done.set()

        directions.calculateDirectionsWithCompletionHandler_(on_calc)

    geocoder.geocodeAddressString_completionHandler_(destination, on_dest_geocoded)
    done.wait(timeout=20)

    if result["error"]:
        raise RuntimeError(str(result["error"]))
    if not result["steps"]:
        raise RuntimeError("Timed out while requesting directions.")

    data = result["steps"]
    steps = data["steps"]
    lines = [
        f"Walking directions to {destination}:",
        f"Total distance: about {data['total_feet']} feet",
        f"Estimated time: about {data['total_minutes']} minutes",
    ]
    for idx, step in enumerate(steps, 1):
        lines.append(f"Step {idx}: {step}")
    return "\n".join(lines)


def _get_directions_fallback(destination: str, origin: str | None = None) -> str:
    origin_text = f" from {origin}" if origin else ""
    return (
        f"I couldn't access map directions right now. I can still help you orient to nearby "
        f"landmarks while you head toward {destination}{origin_text}."
    )


if __name__ == "__main__":
    print(get_directions("Starbucks", "500 Howard St San Francisco"))
